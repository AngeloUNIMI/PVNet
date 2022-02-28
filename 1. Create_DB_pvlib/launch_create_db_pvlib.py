import pvlib
import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import unicodedata
import re
import math


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


if __name__ == "__main__":

    # downsample
    dsModules = 10
    dsInverters = 10

    # temperature parameters
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # latitude, longitude, name, altitude, timezone
    coordinates = [
        (55.94, -3.14, 'Edinburgh', 70, 'GMT'),
        (45.46, 9.19, 'Milan', 134, 'GMT+1'),
        # (45.05, 9.68, 'Piacenza', 63, 'GMT+1'),
        # (38.11, 13.35, 'Palermo', 30, 'GMT+1'),
        # (25.26, 55.31, 'Dubai', 2, 'GMT+4'),
    ]

    # years = [2014, 2015, 2016, 2017, 2018, 2019]
    years = range(2000, 2020)

    # loop on locations
    for location in coordinates:

        # display
        print(location)

        latitude, longitude, name, altitude, timezone = location
        weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude, map_variables=True)[0]
        weather.index.name = "utc_time"
        # tmys = []
        # tmys.append(weather)

        # extract modules
        sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        # downsample
        numModules = sandia_modules.shape[1]
        indexModules = range(0, numModules, math.floor(numModules/dsModules))
        sandia_modules = sandia_modules[sandia_modules.columns[indexModules]]
        # loop on modules
        # module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        for module_count, module_name in enumerate(sandia_modules):

            # display
            print('\tModule ' + (module_count+1).__str__() + ': ' + module_name)

            # sanitize
            module_name_slug = slugify(module_name)

            # ------------------
            # if module_count > 3:
            # break
            # ------------------

            # extract module
            module = sandia_modules[module_name]

            # extract inverters
            sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
            # downsample
            numInverters = sapm_inverters.shape[1]
            indexInverters = range(0, numInverters, math.floor(numInverters/dsInverters))
            sapm_inverters = sapm_inverters[sapm_inverters.columns[indexInverters]]
            # loop on inverters
            # inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
            for inverter_count, inverter_name in enumerate(sapm_inverters):

                # display
                # print('\t\t' + inverter_name)

                # sanitize
                inverter_name_slug = slugify(inverter_name)

                # ------------------
                # if inverter_count > 0:
                # break
                # ------------------

                # extract inverter
                inverter = sapm_inverters[inverter_name]

                # create folder
                folder = './data_db/' + name + '/' + module_name_slug + '/' + inverter_name_slug + '/'
                os.makedirs(folder, exist_ok=True)

                # print('\t\t\t', end='')

                # loop on years
                for year_single in years:

                    # display
                    # print(year_single.__str__() + '\t', end='')

                    # filename
                    filename = folder + year_single.__str__() + '.dat'

                    if os.path.isfile(filename):
                        continue

                    # year
                    date_start = year_single.__str__() + '-01-01'
                    date_end = year_single.__str__() + '-12-31'
                    weather_year = weather[(weather.index >= date_start) & (weather.index <= date_end)]

                    # system specification
                    system = {'module': module, 'inverter': inverter, 'surface_azimuth': 180}

                    # solar position
                    latitude, longitude, name, altitude, timezone = location
                    system['surface_tilt'] = latitude
                    solpos = pvlib.solarposition.get_solarposition(
                        time=weather_year.index,
                        latitude=latitude,
                        longitude=longitude,
                        altitude=altitude,
                        temperature=weather_year["temp_air"],
                        pressure=pvlib.atmosphere.alt2pres(altitude),)

                    # radiation
                    dni_extra = pvlib.irradiance.get_extra_radiation(weather_year.index)
                    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
                    pressure = pvlib.atmosphere.alt2pres(altitude)
                    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
                    aoi = pvlib.irradiance.aoi(
                        system['surface_tilt'],
                        system['surface_azimuth'],
                        solpos["apparent_zenith"],
                        solpos["azimuth"],)

                    # irradiance
                    total_irradiance = pvlib.irradiance.get_total_irradiance(
                        system['surface_tilt'],
                        system['surface_azimuth'],
                        solpos['apparent_zenith'],
                        solpos['azimuth'],
                        weather_year['dni'],
                        weather_year['ghi'],
                        weather_year['dhi'],
                        dni_extra=dni_extra,
                        model='haydavies',)

                    # temperature
                    cell_temperature = pvlib.temperature.sapm_cell(
                        total_irradiance['poa_global'],
                        weather_year["temp_air"],
                        weather_year["wind_speed"],
                        **temperature_model_parameters,)

                    # effective irradiance
                    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
                        total_irradiance['poa_direct'],
                        total_irradiance['poa_diffuse'],
                        am_abs,
                        aoi,
                        module,)

                    # dc, ac
                    dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
                    ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)
                    annual_energy = ac.sum()

                    # save
                    file = open(filename, 'wb')
                    dataSave = {'latitude': latitude, 'longitude': longitude,
                                'name': name, 'altitude': altitude,
                                'timezone': timezone, 'weather': weather_year,
                                'dc': dc, 'ac': ac, 'annual_energy': annual_energy,
                                'module_name': module_name, 'inverter_name': inverter_name}
                    pickle.dump(dataSave, file)
                    file.close()

                # print('')
