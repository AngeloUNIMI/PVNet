import os
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from random import seed
from random import random
from datetime import datetime
import pickle
import math
import unicodedata
import re
from sklearn.model_selection import KFold
# user-defined
import util


if __name__ == "__main__":

    # params
    plotta = False
    log = True


    # -------------------------------------------------------------------
    dirWorkspace = '../1. Create_DB_pvlib'
    dbname = 'data_db'
    dbfolder = dirWorkspace + '/' + dbname + '/'

    # latitude, longitude, name, altitude, timezone
    # coordinates = [
        # (55.94, -3.14, 'Edinburgo', 70, 'GMT'),
        # (45.46, 9.19, 'Milano', 134, 'GMT+1'),
        # (45.05, 9.68, 'Piacenza', 63, 'GMT+1'),
        # (38.11, 13.35, 'Palermo', 30, 'GMT+1'),
        # (25.26, 55.31, 'Dubai', 2, 'GMT+4'),
    # ]

    # years = [2014, 2015, 2016, 2017, 2018, 2019]
    years = range(2000, 2020)

    # system spec common to all
    surface_azimuth = 180
    # temperature parameters
    temperature_model_parametersTrain = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
    temperature_model_parametersTest = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # list modules
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    # list inverters
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    # list locations
    folder_cities = [name for name in os.listdir(dbfolder)
                     if os.path.isdir(os.path.join(dbfolder, name))]

    # loop on locations
    for count_city, city in enumerate(folder_cities):

        # expand location
        # latitude, longitude, name, altitude, timezone = location

        # process folders
        folderLocation = os.path.join(dbfolder, city)
        # results
        folderResults = './features_' + dbname + '/' + city + '/'
        os.makedirs(folderResults, exist_ok=True)
        # result file
        now = datetime.now()
        current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
        fileResultName = current_time + '.txt'
        fileResultNameFull = os.path.join(folderResults, fileResultName)
        fileResult = open(fileResultNameFull, "x")
        fileResult.close()

        # display
        util.print_pers(city, fileResultNameFull)

        # list subfolders - types of PV cells
        folderPVcells = [name for name in os.listdir(folderLocation)
                         if os.path.isdir(os.path.join(folderLocation, name))]
        # number of pv cells
        numPVcells = len(folderPVcells)

        # -------------------------------------------------
        # EXTRACT FEATURES
        for module_count, folderModule in enumerate(folderPVcells):
            folderEachPVCell = os.path.join(folderLocation, folderModule)

            # select PV cell
            # moduleTrain = sandia_modules[folderPVcells[indexTrain]]
            # moduleTrain = util.selectModule(folderPVcells[indexTrain], sandia_modules)

            # display
            # util.print_pers('\t' + folderModule, fileResultNameFull)
            util.print_pers('\tModule ' + (module_count+1).__str__() + ': ' + folderModule, fileResultNameFull)

            # list folders with INV types
            folderINVtypes = [name for name in os.listdir(folderEachPVCell)
                              if os.path.isdir(os.path.join(folderEachPVCell, name))]

            # loop on INV folders
            for inv_count, eachINVfolder in enumerate(folderINVtypes):

                # folder for the specific INV type
                folderEachINVtype = os.path.join(folderEachPVCell, eachINVfolder)

                # select inverter
                # inverterTrain = sapm_inverters[eachINVfolder]
                # inverterTrain = util.selectInverter(eachINVfolder, sapm_inverters)

                # display
                # util.print_pers('\t\t' + eachINVfolder, fileResultNameFull)
                # util.print_pers('\t\tInverter ' + (inv_count+1).__str__() + ': ' + eachINVfolder, fileResultNameFull)

                # print('\t\t\t', end='')

                # list year files
                filesYears = os.listdir(folderEachINVtype)
                # init combined
                weatherAllYears = pd.DataFrame()
                # loop on year files
                for year_count, fileYearSingle in enumerate(filesYears):
                    # check it is a .dat file
                    if fileYearSingle.endswith('.dat'):

                        YearSingle = os.path.splitext(fileYearSingle)[0]

                        # display
                        # print(fileYearSingle + ' ', end='')

                        filePath = os.path.join(folderEachINVtype, fileYearSingle)

                        # load
                        file = open(filePath, 'rb')
                        dataYear = pickle.load(file)
                        file.close()

                        # load module_name and inverter
                        if year_count == 0:
                            # select PV cell
                            module = sandia_modules[dataYear['module_name']]
                            # select inverter
                            inverter = sapm_inverters[dataYear['inverter_name']]

                        # load location info
                        # if count_city == 0:
                        latitude = dataYear['latitude']
                        longitude = dataYear['longitude']
                        altitude = dataYear['altitude']
                        timezone = dataYear['timezone']

                        # concatenate years
                        # frames = [weatherAllYears, dataYear['weather']]
                        # weatherAllYears = pd.concat(frames)
                        weatherSingleYear = dataYear['weather']
                        numSamplesWeather = weatherSingleYear.shape[0]
                        if numSamplesWeather == 0:
                            continue
                        # print()

                        # ------------------------------------
                        # extract features
                        featuresSingle, \
                        surface_tilt, solpos, dni_extra, airmass, pressure, \
                        am_abs, aoi, total_irradiance, cell_temperature, \
                        effective_irradiance = util.extractFeatures(latitude, longitude, altitude,
                                                                    weatherSingleYear, surface_azimuth,
                                                               temperature_model_parametersTrain, module)
                        # extract output
                        targetSingle, \
                        dc, ac, annual_energy = util.extractTarget(effective_irradiance, cell_temperature,
                                                              module, inverter)
                        # update features for all PV and INV types
                        # frames = [features, featuresSingle]
                        # features = pd.concat(frames, axis=0)
                        # update output for all PV and INV types
                        # frames = [target, outputSingle]
                        # target = pd.concat(frames, axis=0)

                        # combine feature and target in one dataframe
                        frames = [featuresSingle, targetSingle]
                        allData = pd.concat(frames, axis=1)

                        # basic preprocessing
                        allData = util.basicPreproc(allData)

                        # size of features
                        numSamples = allData.shape[0]
                        numFeatures = featuresSingle.shape[1]
                        # util.print_pers('\tNum samples: {0}'.format(numSamples), fileResultNameFull)
                        # util.print_pers('\tNum features: {0}'.format(numFeatures), fileResultNameFull)

                        # save
                        fileResultFeatures = YearSingle + '.dat'
                        folderPickleDump = os.path.join(folderResults, folderModule, eachINVfolder)
                        os.makedirs(folderPickleDump, exist_ok=True)
                        fileResultFeaturesFull = os.path.join(folderPickleDump, fileResultFeatures)
                        file = open(fileResultFeaturesFull, 'wb')
                        dataSave = {
                            'latitude': latitude, 'longitude': longitude,
                            'name': city, 'altitude': altitude,
                            'timezone': timezone,
                            # 'weather': weatherAllYears,
                            'allData': allData,
                            'numSamples': numSamples, 'numFeatures': numFeatures}
                        pickle.dump(dataSave, file)
                        file.close()



