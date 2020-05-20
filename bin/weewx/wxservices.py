#
#    Copyright (c) 2009-2020 Tom Keffer <tkeffer@gmail.com>
#
#    See the file LICENSE.txt for your full rights.
#

"""
This module performs two functions:
1. Adds weather-related extensions to the WeeWX type system.
2. Uses those extensions to augment packets and records with derived types.
"""

from __future__ import absolute_import
from __future__ import print_function

import logging
from configobj import ConfigObj
from six import iteritems

import weedb
import weeutil.logger
import weeutil.weeutil
import weewx.engine
import weewx.manager
import weewx.units
import weewx.wxformulas
import weewx.xtypes
from six.moves import StringIO
from weeutil.weeutil import to_int, to_float, to_bool
from weewx.units import ValueTuple, mps_to_mph, kph_to_mph, METER_PER_FOOT, CtoF

log = logging.getLogger(__name__)

# TODO. Defaults are all set for development purposes, to be reduced to no derived obs for release.
DEFAULTS_INI = """
[StdWXCalculate]

    [[Calculations]]
    
        data_binding = wx_binding
        ignore_zero_wind = True         # If windSpeed is zero, should windDir be set to None?

        [[[Delta]]]
            [[[[rain]]]]
                source = prefer_hardware
                total = yearRain
        [[[Cumulative]]]
            #
        [[[Rater]]]
            [[[[rainRate]]]]
                source = prefer_hardware
                source_field = rain
                period = 900                # Rain rate window
                retain_period = 930         # How long to retain rain events. Should be >= rain_period + archive_delay
        [[[PressureCooker]]]
            [[[[pressure]]]]
                source = prefer_hardware
                max_delta_12h = 1800        # When looking up a temperature in the past, how close does the time have to be?
            [[[[altimeter]]]]
                source = prefer_hardware
                algorithm = aaASOS
            [[[[barometer]]]]
                source = prefer_hardware
        [[[WXXTypes]]]
            [[[[appTemp]]]]
                source = prefer_hardware
            [[[[beaufort]]]]
                source = prefer_hardware
            [[[[cloudbase]]]]
                source = prefer_hardware
            [[[[dewpoint]]]]
                source = prefer_hardware
            [[[[ET]]]]
                source = prefer_hardware
                et_period = 3600            # For evapotranspiration
                wind_height = 2.0, meter    # For evapotranspiration. In meters.
            [[[[heatindex]]]]
                source = prefer_hardware
            [[[[humidex]]]]
                source = prefer_hardware
            [[[[inDewpoint]]]]
                source = prefer_hardware
            [[[[maxSolarRad]]]]
                source = prefer_hardware
                algorithm = RS
                atc = 0.8                   # For solar radiation RS
                nfac = 2                    # Atmospheric turbidity (2=clear, 4-5=smoggy)
            [[[[windchill]]]]
                source = prefer_hardware
            [[[[windrun]]]]
                source= prefer_hardware
"""


class StdWXCalculate(weewx.engine.StdService):
    """Wrapper class to allow WXCalculate to be used as a WeeWX service"""

    def __init__(self, engine, config_dict):
        """Initialize the service."""
        super(StdWXCalculate, self).__init__(engine, config_dict)

        # Instantiate a WXCalculate object to do the heavy work
        self.calc = WXCalculate(config_dict,
                                engine.stn_info.altitude_vt,
                                engine.stn_info.latitude_f,
                                engine.stn_info.longitude_f,
                                engine.db_binder)

        # we will process both loop and archive events
        self.bind(weewx.NEW_LOOP_PACKET, self.new_loop_packet)
        self.bind(weewx.NEW_ARCHIVE_RECORD, self.new_archive_record)

    def new_loop_packet(self, event):
        self.calc.new_loop_packet(event.packet)

        # Now augment the packet with extended types as per the configuration
        self.calc.do_calculations(event.packet, 'loop')

    def new_archive_record(self, event):
        self.calc.do_calculations(event.record, 'archive')

    def shutDown(self):
        self.calc.shut_down()


class WXCalculate(object):
    """This class has two jobs:

    - Add derived weather variables (such as dewpoint, heatindex, etc.) to the WeeWX extensible
    type system.
    - Use the type system to augment packets and records, following preferences specified in the
    configuration file.
    """

    def __init__(self, config_dict, altitude_vt, latitude_f, longitude_f, db_binder=None):
        """Initialize the service."""

        # Start with the default configuration. Make a copy --- we will be modifying it
        merge_dict = ConfigObj(StringIO(DEFAULTS_INI), encoding='utf-8')
        # Now merge in the overrides from the config file
        merge_dict.merge(config_dict)
        # Extract out the part we're interested in
        self.svc_dict = merge_dict['StdWXCalculate']

        if db_binder is None:
            db_binder = weewx.manager.DBBinder(config_dict)
        self.db_manager = db_binder.get_manager(
            data_binding=self.svc_dict.get('data_binding', 'wx_binding'),
            initialize=True)

        self.ignore_zero_wind = to_bool(self.svc_dict.get('ignore_zero_wind', True))

        self.altitude_vt = altitude_vt
        self.latitude = latitude_f
        self.longitude = longitude_f
        self.loop_call_list = list()
        for section in self.svc_dict.sections:
            xtype_obj_list = self.xtype_factory(section)
            if xtype_obj_list:
                weewx.xtypes.xtypes += xtype_obj_list

        # Report about which values will be calculated...
        log.info("The following values will be calculated: %s",
                 ', '.join(["%s=%s" % (k, v['source'])
                            for k, v in iteritems(self.svc_dict) if hasattr(v, 'keys')]))

        # ...and which algorithms will be used.
        log.info("The following algorithms will be used for calculations: %s",
                 ', '.join(["%s=%s" % (k, v['algorithm'])
                            for k, v in iteritems(self.svc_dict) if
                            hasattr(v, 'keys') and 'algorithm' in v]))

    def xtype_factory(self, section):
        """Factory method to create XType objects to support derived obs."""

        section_config = self.svc_dict[section]
        if len(section_config) > 0:
            instantiate_method = '_'.join(['instantiate', section.lower()])
            return getattr(self, instantiate_method)(section_config)
        return list()

    def instantiate_delta(self, config_dict):
        """Instantiate any Delta objects."""

        object_list = []
        for delta_sect in config_dict:
            object_list.append(Delta(delta_sect, **config_dict[delta_sect]))
        return object_list

    def instantiate_cumulative(self, config_dict):
        """Instantiate any Cumulative objects."""

        object_list = []
        for cumulative_sect in config_dict:
            object_list.append(Cumulative(cumulative_sect,
                                          **config_dict[cumulative_sect]))
        return object_list

    def instantiate_pressurecooker(self, config_dict):
        """Instantiate any PressureCooker objects."""

        object = PressureCooker(config_dict,
                                altitude_vt=self.altitude_vt)
        return [object, ]

    def instantiate_wxxtypes(self, config_dict):
        """Instantiate any WXXTypes objects."""

        object = WXXTypes(config_dict,
                          altitude_vt=self.altitude_vt,
                          latitude=self.latitude,
                          longitude=self.longitude)
        return [object, ]

    def instantiate_rater(self, config_dict):
        """Instantiate any Rater objects."""

        object_list = []
        for rater_sect in config_dict:
            object = Rater(rater_sect, **config_dict[rater_sect])
            object_list.append(object)
            self.loop_call_list.append({'object': object,
                                        'method': 'add_loop_packet'})
        return object_list

    def new_loop_packet(self, loop_packet):

        # Keep any Raters up to date:
        for call in self.loop_call_list:
            call['object'].call['method'](loop_packet, self.db_manager)

    def do_calculations(self, data_dict, data_type):
        """Augment the data dictionary with derived types as necessary.

        data_dict: The incoming LOOP packet or archive record.

        data_type: = "loop" if LOOP packet;
                   = "record" if archive record.
        """
        if self.ignore_zero_wind:
            self.adjust_winddir(data_dict)

        # Go through the list of potential calculations and see which ones need
        # to be done
        for derived_obs in self.svc_dict.sections:
            # Get the 'source' for the obs
            directive = self.svc_dict[derived_obs].get('source')
            # Keys in svc_dict are in unicode. Keys in packets and records
            # are in native strings. Just to keep things consistent,
            # convert.
            obs_type = str(derived_obs)
            if directive == 'software' or directive == 'prefer_hardware' \
                    and (obs_type not in data_dict or data_dict[obs_type] is None):
                try:
                    # We need to do a calculation for type 'obs_type'. This
                    # may raise an exception.
                    new_value = weewx.xtypes.get_scalar(obs_type,
                                                        data_dict,
                                                        self.db_manager)
                except weewx.CannotCalculate:
                    pass
                except weewx.UnknownType as e:
                    log.debug("Unknown extensible type '%s'" % e)
                except weewx.UnknownAggregation as e:
                    log.debug("Unknown aggregation '%s'" % e)
                else:
                    # If there was no exception, add the results to the
                    # dictionary
                    data_dict[obs_type] = new_value[0]

    @staticmethod
    def adjust_winddir(data):
        """If windSpeed is in the data stream, and it is either zero or None, then the
        wind direction is undefined.
        """
        if 'windSpeed' in data and not data['windSpeed']:
            data['windDir'] = None
        if 'windGust' in data and not data['windGust']:
            data['windGustDir'] = None

    def shut_down(self):
        for xtype in ['pressure_cooker', 'rain_maker',
                      'rain_rater', 'wx_types']:
            if hasattr(self, xtype):
                # Give the object an opportunity to clean up
                getattr(self, type).shut_down()
                # Remove from the type system
                weewx.xtypes.xtypes.remove(getattr(self, type))
        self.db_manager = None


class WXXTypes(weewx.xtypes.XType):
    """Weather extensions to the WeeWX type extension system that are relatively simple. This is
    for types which are generally stateless, such as dewpoint, heatindex, etc.
    """

    def __init__(self, svc_dict, altitude_vt, latitude, longitude):
        """Initialize an instance of WXXTypes

        Args:
            svc_dict: ConfigDict structure with configuration info
            altitude_vt: The altitude of the station as a ValueTuple
            latitude:  Its latitude
            longitude:  Its longitude
        """

        self.svc_dict = svc_dict
        self.altitude_vt = altitude_vt
        self.latitude = latitude
        self.longitude = longitude

        # ET specific properties
        if 'ET' in svc_dict:
            # Window of time for evapotranspiration calculation, in seconds
            self.et_period = to_int(svc_dict['ET'].get('et_period', 3600))
            # Height above ground at which wind is measured, in meters
            wind_height = weeutil.weeutil.option_as_list(svc_dict['ET'].get('wind_height',
                                                                            (2.0, 'meter')))
            # wind_height may be specified as a single value in meters or as
            # a value, unit pair
            if len(wind_height) > 1:
                # We have a value, unit pair so we need to do any necessary
                # conversion to meters. First construct a ValueTuple.
                wh_vt = ValueTuple(float(wind_height[0]), wind_height[1], 'group_altitude')
                # Then convert to meters
                self.wind_height = weewx.units.convert(wh_vt, 'meter').value
            else:
                # We have a value only so assume it is meters
                self.wind_height = float(wind_height[0])
        else:
            self.et_period = 3600
            self.wind_height = 2.0
        # maxSolarRad specific properties
        if 'maxSolarRad' in svc_dict:
            # atmospheric transmission coefficient [0.7-0.91]
            self.atc = to_float(svc_dict['maxSolarRad'].get('atc', 0.8))
            # Fail hard if out of range:
            if not 0.7 <= self.atc <= 0.91:
                raise weewx.ViolatedPrecondition("Atmospheric transmission "
                                                 "coefficient (%f) out of "
                                                 "range [.7-.91]" % self.atc)
            # atmospheric turbidity (2=clear, 4-5=smoggy)
            self.nfac = to_float(svc_dict.get('nfac', 2))
        else:
            self.atc = 0.8
            self.nfac = 2

    def get_scalar(self, obs_type, record, db_manager):

        # Get the method name for this observation type
        method_name = 'calc_%s' % obs_type
        try:
            # Now call it with arguments
            return getattr(self, method_name)(obs_type, record, db_manager)
        except AttributeError:
            raise weewx.UnknownType(obs_type)

    def calc_maxSolarRad(self, key, data, db_manager):
        try:
            algo = self.svc_dict['Algorithms']['maxSolarRad'].lower()
        except KeyError:
            algo = 'rs'
        altitude_m = weewx.units.convert(self.altitude_vt, 'meter')[0]
        if algo == 'bras':
            val = weewx.wxformulas.solar_rad_Bras(self.latitude, self.longitude, altitude_m,
                                                  data['dateTime'], self.nfac)
        elif algo == 'rs':
            val = weewx.wxformulas.solar_rad_RS(self.latitude, self.longitude, altitude_m,
                                                data['dateTime'], self.atc)
        else:
            raise weewx.ViolatedPrecondition("Unknown solar algorithm '%s'"
                                             % self.svc_dict['Algorithms']['maxSolarRad'])
        return ValueTuple(val, 'watt_per_meter_squared', 'group_radiation')

    def calc_cloudbase(self, key, data, db_manager):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        # Convert altitude to the same unit system as the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, data['usUnits'])
        # Use the appropriate formula
        if data['usUnits'] == weewx.US:
            formula = weewx.wxformulas.cloudbase_US
            u = 'foot'
        else:
            formula = weewx.wxformulas.cloudbase_Metric
            u = 'meter'
        val = formula(data['outTemp'], data['outHumidity'], altitude[0])
        return ValueTuple(val, u, 'group_altitude')

    def calc_ET(self, key, data, db_manager):
        """Get maximum and minimum temperatures and average radiation and wind speed for the
        indicated period then calculate the amount of evapotranspiration during the interval.
        Convert to US units if necessary since this service operates in US unit system.
        """

        if 'interval' not in data:
            # This will cause LOOP data not to be processed.
            raise weewx.CannotCalculate(key)

        interval = data['interval']
        end_ts = data['dateTime']
        start_ts = end_ts - self.et_period
        try:
            r = db_manager.getSql("SELECT MAX(outTemp), MIN(outTemp), "
                                  "AVG(radiation), AVG(windSpeed), "
                                  "MAX(outHumidity), MIN(outHumidity), "
                                  "MAX(usUnits), MIN(usUnits) FROM %s "
                                  "WHERE dateTime>? AND dateTime <=?"
                                  % db_manager.table_name, (start_ts, end_ts))
        except weedb.DatabaseError:
            return ValueTuple(None, None, None)

        # Make sure everything is there:
        if r is None or None in r:
            return ValueTuple(None, None, None)

        # Unpack the results
        T_max, T_min, rad_avg, wind_avg, rh_max, rh_min, std_unit_min, std_unit_max = r

        # Check for mixed units
        if std_unit_min != std_unit_max:
            log.info("Mixed unit system not allowed in ET calculation. Skipped.")
            return ValueTuple(None, None, None)
        std_unit = std_unit_min
        if std_unit == weewx.METRIC or std_unit == weewx.METRICWX:
            T_max = CtoF(T_max)
            T_min = CtoF(T_min)
            if std_unit == weewx.METRICWX:
                wind_avg = mps_to_mph(wind_avg)
            else:
                wind_avg = kph_to_mph(wind_avg)
        # Wind height is in meters, so convert it:
        height_ft = self.wind_height / METER_PER_FOOT
        # Get altitude in feet
        altitude_ft = weewx.units.convert(self.altitude_vt, 'foot')[0]

        try:
            ET_rate = weewx.wxformulas.evapotranspiration_US(
                T_min, T_max, rh_min, rh_max, rad_avg, wind_avg, height_ft,
                self.latitude, self.longitude, altitude_ft, end_ts)
        except ValueError as e:
            log.error("Calculation of evapotranspiration failed: %s", e)
            weeutil.logger.log_traceback(log.error)
            ET_inch = None
        else:
            # The formula returns inches/hour. We need the total ET over the interval, so multiply
            # by the length of the interval in hours. Remember that 'interval' is actually in
            # minutes.
            ET_inch = ET_rate * interval / 60.0 if ET_rate is not None else None

        # Convert back to the unit system of the incoming record:
        ET = weewx.units.convertStd((ET_inch, 'inch', 'group_rain'), data['usUnits'])
        return ET

    @staticmethod
    def calc_dewpoint(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.dewpointF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.dewpointC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_inDewpoint(key, data, db_manager=None):
        if 'inTemp' not in data or 'inHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.dewpointF(data['inTemp'], data['inHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.dewpointC(data['inTemp'], data['inHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_windchill(key, data, db_manager=None):
        if 'outTemp' not in data or 'windSpeed' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.windchillF(data['outTemp'], data['windSpeed'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.windchillC(data['outTemp'], data['windSpeed'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_heatindex(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.heatindexF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.heatindexC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_humidex(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.humidexF(data['outTemp'], data['outHumidity'])
            u = 'degree_F'
        else:
            val = weewx.wxformulas.humidexC(data['outTemp'], data['outHumidity'])
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_appTemp(key, data, db_manager=None):
        if 'outTemp' not in data or 'outHumidity' not in data or 'windSpeed' not in data:
            raise weewx.CannotCalculate(key)
        if data['usUnits'] == weewx.US:
            val = weewx.wxformulas.apptempF(data['outTemp'], data['outHumidity'],
                                            data['windSpeed'])
            u = 'degree_F'
        else:
            # The metric equivalent needs wind speed in mps. Convert.
            windspeed_vt = weewx.units.as_value_tuple(data, 'windSpeed')
            windspeed_mps = weewx.units.convert(windspeed_vt, 'meter_per_second')[0]
            val = weewx.wxformulas.apptempC(data['outTemp'], data['outHumidity'], windspeed_mps)
            u = 'degree_C'
        return weewx.units.convertStd((val, u, 'group_temperature'), data['usUnits'])

    @staticmethod
    def calc_beaufort(key, data, db_manager=None):
        if 'windSpeed' not in data:
            raise weewx.CannotCalculate
        windspeed_vt = weewx.units.as_value_tuple(data, 'windSpeed')
        windspeed_kn = weewx.units.convert(windspeed_vt, 'knot')[0]
        return ValueTuple(weewx.wxformulas.beaufort(windspeed_kn), None, None)

    @staticmethod
    def calc_windrun(key, data, db_manager=None):
        """Calculate wind run. Requires key 'interval'"""
        if 'windSpeed' not in data or 'interval' not in data:
            raise weewx.CannotCalculate(key)

        if data['windSpeed'] is not None:
            if data['usUnits'] == weewx.US:
                val = data['windSpeed'] * data['interval'] / 60.0
                u = 'mile'
            elif data['usUnits'] == weewx.METRIC:
                val = data['windSpeed'] * data['interval'] / 60.0
                u = 'km'
            elif data['usUnits'] == weewx.METRICWX:
                val = data['windSpeed'] * data['interval'] * 60.0 / 1000.0
                u = 'km'
            else:
                raise weewx.ViolatedPrecondition("Unknown unit system %s" % data['usUnits'])
        else:
            val = None
            u = 'mile'
        return weewx.units.convertStd((val, u, 'group_distance'), data['usUnits'])


class PressureCooker(weewx.xtypes.XType):
    """Pressure related extensions to the WeeWX type system. """

    def __init__(self, svc_dict, altitude_vt)
        # def __init__(self, altitude_vt, max_ts_delta=1800, altimeter_algorithm='aaNOAA'):
        """Initialize the PressureCooker.

        altitude_vt: The altitude as a ValueTuple

        max_ts_delta: When looking up a temperature in the past,
          how close does the time have to be?

        altimeter_algorithm: Algorithm to use to calculate altimeter.
        """

        self.altitude_vt = altitude_vt
        if 'pressure' in svc_dict:
            self.max_ts_delta = int(svc_dict['pressure'].get('max_delta_12h', 1800))
        else:
            self.max_ts_delta = 1800
        if 'altimeter' in svc_dict:
            altimeter_algorithm = svc_dict['altimeter'].get('algorithm', 'aaASOS')
        else:
            altimeter_algorithm = 'aaASOS'
        if not altimeter_algorithm.startswith('aa'):
            altimeter_algorithm = 'aa%s' % altimeter_algorithm
        self.altimeter_algorithm = altimeter_algorithm
        # Timestamp (roughly) 12 hours ago
        self.ts_12h = None
        # Temperature 12 hours ago as a ValueTuple
        self.temp_12h_vt = None

    def _get_temperature_12h(self, ts, dbmanager):
        """Get the temperature as a ValueTuple from 12 hours ago.  The value will
         be None if no temperature is available.
         """

        ts_12h = ts - 12 * 3600

        # Look up the temperature 12h ago if this is the first time through,
        # or we don't have a usable temperature, or the old temperature is too stale.
        if self.ts_12h is None \
                or self.temp_12h_vt is None \
                or abs(self.ts_12h - ts_12h) < self.max_ts_delta:
            # Hit the database to get a newer temperature.
            record = dbmanager.getRecord(ts_12h, max_delta=self.max_ts_delta)
            if record and 'outTemp' in record:
                # Figure out what unit the record is in ...
                unit = weewx.units.getStandardUnitType(record['usUnits'], 'outTemp')
                # ... then form a ValueTuple.
                self.temp_12h_vt = weewx.units.ValueTuple(record['outTemp'], *unit)
            else:
                # Invalidate the temperature ValueTuple from 12h ago
                self.temp_12h_vt = None
            # Save the timestamp
            self.ts_12h = ts_12h

        return self.temp_12h_vt

    def get_scalar(self, key, record, dbmanager):
        if key == 'pressure':
            return self.pressure(record, dbmanager)
        elif key == 'altimeter':
            return self.altimeter(record)
        elif key == 'barometer':
            return self.barometer(record)
        else:
            raise weewx.UnknownType(key)

    def pressure(self, record, dbmanager):
        """Calculate the observation type 'pressure'."""

        # All of the following keys are required:
        if any(key not in record for key in ['usUnits', 'outTemp', 'barometer', 'outHumidity']):
            raise weewx.CannotCalculate('pressure')

        # Get the temperature in Fahrenheit from 12 hours ago
        temp_12h_vt = self._get_temperature_12h(record['dateTime'], dbmanager)
        if temp_12h_vt is None \
                or temp_12h_vt[0] is None \
                or record['outTemp'] is None \
                or record['barometer'] is None \
                or record['outHumidity'] is None:
            pressure = None
        else:
            # The following requires everything to be in US Customary units.
            # Rather than convert the whole record, just convert what we need:
            record_US = weewx.units.to_US({'usUnits': record['usUnits'],
                                           'outTemp': record['outTemp'],
                                           'barometer': record['barometer'],
                                           'outHumidity': record['outHumidity']})
            # Get the altitude in feet
            altitude_ft = weewx.units.convert(self.altitude_vt, "foot")
            # The outside temperature in F.
            temp_12h_F = weewx.units.convert(temp_12h_vt, "degree_F")
            pressure = weewx.uwxutils.uWxUtilsVP.SeaLevelToSensorPressure_12(
                record_US['barometer'],
                altitude_ft[0],
                record_US['outTemp'],
                temp_12h_F[0],
                record_US['outHumidity']
            )

        # Convert to target unit system and return
        return weewx.units.convertStd((pressure, 'inHg', 'group_pressure'), record['usUnits'])

    def altimeter(self, record):
        """Calculate the observation type 'altimeter'."""
        if 'pressure' not in record:
            raise weewx.CannotCalculate('altimeter')

        # Convert altitude to same unit system of the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, record['usUnits'])

        # Figure out which altimeter formula to use, and what unit the results will be in:
        if record['usUnits'] == weewx.US:
            formula = weewx.wxformulas.altimeter_pressure_US
            u = 'inHg'
        else:
            formula = weewx.wxformulas.altimeter_pressure_Metric
            u = 'mbar'
        # Apply the formula
        altimeter = formula(record['pressure'], altitude[0], self.altimeter_algorithm)
        # Convert to the target unit system
        return weewx.units.convertStd((altimeter, u, 'group_pressure'), record['usUnits'])

    def barometer(self, record):
        """Calculate the observation type 'barometer'"""

        if 'pressure' not in record or 'outTemp' not in record:
            raise weewx.CannotCalculate('barometer')

        # Convert altitude to same unit system of the incoming record
        altitude = weewx.units.convertStd(self.altitude_vt, record['usUnits'])

        # Figure out what barometer formula to use:
        if record['usUnits'] == weewx.US:
            formula = weewx.wxformulas.sealevel_pressure_US
            u = 'inHg'
        else:
            formula = weewx.wxformulas.sealevel_pressure_Metric
            u = 'mbar'
        # Apply the formula
        barometer = formula(record['pressure'], altitude[0], record['outTemp'])
        # Convert to the target unit system:
        return weewx.units.convertStd((barometer, u, 'group_pressure'), record['usUnits'])


class Rater(weewx.xtypes.XType):
    """"An extension to the WeeWX type system for calculating rainRate"""

    def __init__(self, field, **rater_dict):
        """Initialize the Rater.

        Args:
            rain_period: The length of the sliding window in seconds.
            retain_period: How long to retain a rain event. Should be rain_period
              plus archive_delay.
        """
        self.field = field
        self.source_field = rater_dict.get('source_field')
        self.period = to_int(rater_dict.get('period', 900))
        self.retain_period = to_int(rater_dict.get('retain_period', 930))
        self.events = None
        self.unit_system = None

    def add_loop_packet(self, record, db_manager):
        # Was there any rain? If so, convert the rain to the unit system we are using,
        # then intern it
        if 'rain' in record and record['rain']:
            if self.unit_system is None:
                # Adopt the unit system of the first record.
                self.unit_system = record['usUnits']
            if self.events is None:
                self._setup(record['dateTime'], db_manager)
            # Get the unit system and group of the incoming rain. In theory, this should be
            # the same as self.unit_system, but ...
            u, g = weewx.units.getStandardUnitType(record['usUnits'], self.source_field)
            # Convert to the unit system that we are using
            rain = weewx.units.convertStd((record[self.source_field], u, g), self.unit_system)[0]
            # Add it to the list of events
            self.events.append((record['dateTime'], rain))

        if self.events:
            # Trim any old packets:
            self.events = [x for x in self.events
                           if x[0] >= record['dateTime'] - self.period]

    def get_scalar(self, key, record, db_manager):
        """Calculate the Rate"""
        if key != self.field:
            raise weewx.UnknownType(key)

        if self.events is None:
            self._setup(record['dateTime'], db_manager)

        # Sum the rain events within the time window...
        rainsum = sum(x[1] for x in self.events
                      if x[0] > record['dateTime'] - self.period)
        # ...then divide by the period and scale to an hour
        val = 3600 * rainsum / self.period
        # Get the unit and unit group for rainRate
        u, g = weewx.units.getStandardUnitType(self.unit_system, self.field)
        # Form a ValueTuple, then convert it to the unit system of the incoming record
        rr = weewx.units.convertStd(ValueTuple(val, u, g), record['usUnits'])
        return rr

    def _setup(self, stop_ts, db_manager):
        """Initialize the event list"""
        if self.events is None:
            self.events = []
        start_ts = stop_ts - self.retain_period
        # Get all events since the window start from the database. Put it in
        # a 'try' block because the database may not have a 'rain' field.
        try:
            for row in db_manager.genSql("SELECT dateTime, usUnits, %s FROM %s "
                                         "WHERE dateTime>? AND dateTime<=?;"
                                         % self.source_field, db_manager.table_name,
                                         (start_ts, stop_ts)):
                # Unpack the row:
                time_ts, unit_system, rain = row
                self.add_loop_packet(
                    {'dateTime': time_ts, 'usUnits': unit_system, self.source_field: rain},
                    db_manager)
        except weedb.DatabaseError as e:
            log.debug("Database error while initializing %s: '%s'" % (self.field, e))


class Delta(weewx.xtypes.XType):
    """"An extension to the WeeWX type system for calculating rain"""

    def __init__(self, field, **delta_dict):
        """Initialize the RainMaker.

        Args:
            source_field: Source field from which rain is derived.
        """
        self.field = field
        self.source_field = delta_dict.get('source_field')
        # Initialise the last source field value seen
        self.last_value = None

    def get_scalar(self, key, record, db_manager):
        """Calculate rain"""
        if key != self.field:
            raise weewx.UnknownType(key)

        if self.source_field is not None and self.source_field in record:
            # Calculate the rain value
            _delta = weewx.wxformulas.calculate_rain(record[self.source_field],
                                                     self.last_value)
            # Update the last source field value seen
            self.last_value = record[self.source_field]
            unit, group = weewx.units.getStandardUnitType(record['usUnits'], self.field)
            # Return the rain value as a ValueTuple
            return ValueTuple(_delta, unit, group)
        # If the source field is not available we cannot calculate rain so
        # raise a CannotCalculate exception
        raise weewx.CannotCalculate(self.field)


class Cumulative(weewx.xtypes.XType):
    """"An extension to the WeeWX type system for calculating rain"""

    def __init__(self, field, **cumulative_dict):
        """Initialize the RainMaker.

        Args:
            source_field: Source field from which rain is derived.
        """
        pass

    def get_scalar(self, key, record, db_manager):
        """Calculate rain"""
        raise weewx.CannotCalculate(self.field)
