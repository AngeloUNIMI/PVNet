import pvlib
import pandas as pd


def extractFeatures(latitude, longitude, altitude, weatherAllYears, surface_azimuth,
                    temperature_model_parameters, module):

    # RELATED ONLY TO WEATHER AND POSITION
    # solar position
    surface_tilt = latitude
    solpos = pvlib.solarposition.get_solarposition(
        time=weatherAllYears.index,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        temperature=weatherAllYears["temp_air"],
        pressure=pvlib.atmosphere.alt2pres(altitude),)

    # radiation
    dni_extra = pvlib.irradiance.get_extra_radiation(weatherAllYears.index)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    aoi = pvlib.irradiance.aoi(
        surface_tilt,
        surface_azimuth,
        solpos["apparent_zenith"],
        solpos["azimuth"],)

    # irradiance
    total_irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt,
        surface_azimuth,
        solpos['apparent_zenith'],
        solpos['azimuth'],
        weatherAllYears['dni'],
        weatherAllYears['ghi'],
        weatherAllYears['dhi'],
        dni_extra=dni_extra,
        model='haydavies',)

    # temperature
    cell_temperature = pvlib.temperature.sapm_cell(
        total_irradiance['poa_global'],
        weatherAllYears["temp_air"],
        weatherAllYears["wind_speed"],
        **temperature_model_parameters,)


    # RELATED TO MODULE AND INVERTER
    # effective irradiance
    effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
        total_irradiance['poa_direct'],
        total_irradiance['poa_diffuse'],
        am_abs,
        aoi,
        module,)


    # ------------------------------------
    # features: output related to weather
    # surface_tilt, solpos, dni_extra, airmass, pressure,
    # am_abs, aoi, total_irradiance, cell_temperature
    rows, columns = weatherAllYears.shape
    surface_tilt = pd.Series(surface_tilt).repeat(rows)
    surface_tilt.index = weatherAllYears.index
    surface_tilt.name = 'surface_tilt'
    pressure = pd.Series(pressure).repeat(rows)
    pressure.index = weatherAllYears.index
    pressure.name = 'pressure'

    # update features
    features = pd.DataFrame()
    frames = [features,
              weatherAllYears,
              surface_tilt, solpos, dni_extra, airmass, pressure,
              am_abs, aoi, total_irradiance, cell_temperature,
              #effective_irradiance
              ]
    features = pd.concat(frames, axis=1)

    # return
    return features, \
           surface_tilt, solpos, dni_extra, airmass, pressure, \
           am_abs, aoi, total_irradiance, cell_temperature, \
           effective_irradiance