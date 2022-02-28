import os
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from random import seed
from random import random
from datetime import datetime, date, timedelta
import pickle
import math
from util.train_model import err_mae
import unicodedata
import re
from sklearn.model_selection import KFold
import calendar

# user-defined
import util
from model.model import model_1
from model.model import AngeloNet


if __name__ == "__main__":

    # params
    plotta = False
    log = True
    batch_sizeP = 50  # 32
    batch_sizeP_norm = 50  # 32
    numWorkersP = 0
    num_epochs_train = 10  # 10
    num_epochs_finetune = 10  # 10
    # hiddenNumVector = [3, 5, 10, 20, 30, 50, 100, 200]
    hiddenNumVector = [50]
    dsBatch = 200

    # ------------------------------------------------------------------- Enable CUDA
    cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    Tensor = torch.cuda.DoubleTensor if cuda else torch.cuda.DoubleTensor
    if cuda:
        torch.cuda.empty_cache()
    print("Cuda is {0}".format(cuda))
    print()


    # -------------------------------------------------------------------
    dirWorkspace = '../2. Pvlib_extract_features_v2_MLP'
    dbname = 'data_db'
    dbfolder = dirWorkspace + '/features_' + dbname + '/'

    # years
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

        # loop on num hidden nodes
        for numHidden in hiddenNumVector:

            # process folders
            folderLocation = os.path.join(dbfolder, city)
            # results
            folderResults = './results_' + dbname + '/' + city + '/' + str(numHidden) + '/'
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
            util.print_pers('', fileResultNameFull)

            # display
            util.print_pers('Hidden size: {0}'.format(numHidden), fileResultNameFull)

            # list subfolders - types of PV cells
            folderPVcells = [name for name in os.listdir(folderLocation)
                             if os.path.isdir(os.path.join(folderLocation, name))]
            # number of pv cells
            numPVcells = len(folderPVcells)

            # k fold
            # kf = KFold(n_splits=10, shuffle=True, random_state=1)

            # select 1 pv cell for test, other for train
            errorALL = list()
            errorALL_finetune = list()
            # for i in range(0, 10):
            i = 0
            for module_count, module_name in enumerate(sandia_modules):
            # for kf_train_index, kf_test_index in kf.split(folderPVcells):

                # if exceed numPVcells, exist
                if module_count >= numPVcells:
                    break

                # print("TRAIN:", train_index, "TEST:", test_index)

                # display
                util.print_pers('--------------', fileResultNameFull)
                util.print_pers('Iteration: {0}'.format(i+1), fileResultNameFull)

                # all indexes
                allIndexes = list(range(0, numPVcells))

                #
                testPVcell = [module_count]
                trainPVcell = allIndexes
                trainPVcell.remove(module_count)

                # loop on test year
                for num_testYear, testYear in enumerate(years):

                    # -------------------
                    # if num_testYear > 0:
                        # break

                    # display
                    util.print_pers('Test year: {0}'.format(testYear), fileResultNameFull)

                    # -------------------------------------------------
                    # TEST DATA
                    util.print_pers('TEST DATA - Loading', fileResultNameFull)
                    # init all features
                    allDataTest = pd.DataFrame()
                    # targetTest = pd.DataFrame()
                    # load testing data
                    for num_IndexTest, indexTest in enumerate(testPVcell):

                        # -------------------
                        # if num_IndexTest > 0:
                            # break
                        # -------------------

                        folderEachPVCell = os.path.join(folderLocation, folderPVcells[indexTest])

                        # select PV cell
                        # moduleTest = sandia_modules[folderPVcells[indexTest]]
                        moduleTest = util.selectModule(folderPVcells[indexTest], sandia_modules)

                        # display
                        # util.print_pers('\t' + folderPVcells[indexTest], fileResultNameFull)
                        util.print_pers('\tModule ' + (indexTest + 1).__str__() + ': ' + folderPVcells[indexTest],
                                        fileResultNameFull)

                        # list folders with INV types
                        folderINVtypes = [name for name in os.listdir(folderEachPVCell)
                                          if os.path.isdir(os.path.join(folderEachPVCell, name))]

                        # loop on INV folders
                        for eachINVfolder in folderINVtypes:

                            # folder for the specific INV type
                            folderEachINVtype = os.path.join(folderEachPVCell, eachINVfolder)

                            # select inverter
                            # inverterTest = sapm_inverters[eachINVfolder]
                            inverterTest = util.selectInverter(eachINVfolder, sapm_inverters)

                            # display
                            # util.print_pers('\t\t' + eachINVfolder, fileResultNameFull)
                            # print('\t\t\t', end='')

                            # list year files
                            filesYears = os.listdir(folderEachINVtype)
                            # init combined
                            try:
                                del weatherAllYears, file, dataYear, frames
                            except:
                                pass
                            weatherAllYears = pd.DataFrame()
                            # loop on year files
                            for fileYearSingle in filesYears:

                                # print(fileYearSingle)

                                # check it is a .dat file
                                if fileYearSingle.endswith('.dat'):

                                    # keep test year for testing
                                    YearSingle = os.path.splitext(fileYearSingle)[0]
                                    if YearSingle != str(testYear):
                                        continue

                                    # display
                                    # print(fileYearSingle + ' ', end='')

                                    filePath = os.path.join(folderEachINVtype, fileYearSingle)

                                    # load
                                    file = open(filePath, 'rb')
                                    dataYear = pickle.load(file)
                                    file.close()

                                    # load features and target
                                    allDataTestSingle = dataYear['allData']
                                    # targetSingle = dataYear['target']
                                    numFeaturesTest = dataYear['numFeatures']
                                    # numSamples = dataYear['numSamples']
                                    # concat
                                    frames = [allDataTest, allDataTestSingle]
                                    allDataTest = pd.concat(frames, axis=0)
                                    # frames = [targetTest, targetSingle]
                                    # targetTest = pd.concat(frames, axis=0)

                                    # inutile = 'ciao'
                                    # print(' ', end='')

                    # -------------------------------
                    # break if no samples for testing
                    if (allDataTest.shape[0]) == 0:
                        util.print_pers('', fileResultNameFull)
                        continue
                    # -------------------------------

                    # remove negatives
                    posit = (allDataTest['ac'] >= 0)
                    allDataTest = allDataTest.loc[posit]

                    # remove duplicates
                    allDataTest.drop_duplicates(inplace=True)

                    # -------------------------------------------------
                    # TRAIN DATA
                    util.print_pers('TRAIN DATA - Loading', fileResultNameFull)
                    # init all features
                    allDataTrain = pd.DataFrame()
                    # targetTrain = pd.DataFrame()
                    # load training data
                    for num_indexTrain, indexTrain in enumerate(trainPVcell):
                        folderEachPVCell = os.path.join(folderLocation, folderPVcells[indexTrain])

                        # -------------------
                        # if num_indexTrain > 0:
                            # break
                        # -------------------

                        # select PV cell
                        # moduleTrain = sandia_modules[folderPVcells[indexTrain]]
                        moduleTrain = util.selectModule(folderPVcells[indexTrain], sandia_modules)

                        # display
                        # util.print_pers('\t' + folderPVcells[indexTrain], fileResultNameFull)
                        util.print_pers('\tModule ' + (indexTrain + 1).__str__() + ': ' + folderPVcells[indexTrain],
                                        fileResultNameFull)

                        # list folders with INV types
                        folderINVtypes = [name for name in os.listdir(folderEachPVCell)
                                          if os.path.isdir(os.path.join(folderEachPVCell, name))]

                        # loop on INV folders
                        for eachINVfolder in folderINVtypes:

                            # folder for the specific INV type
                            folderEachINVtype = os.path.join(folderEachPVCell, eachINVfolder)

                            # select inverter
                            # inverterTrain = sapm_inverters[eachINVfolder]
                            inverterTrain = util.selectInverter(eachINVfolder, sapm_inverters)

                            # display
                            # util.print_pers('\t\t' + eachINVfolder, fileResultNameFull)
                            # print('\t\t\t', end='')

                            # list year files
                            filesYears = os.listdir(folderEachINVtype)
                            # init combined
                            weatherAllYears = pd.DataFrame()
                            # loop on year files
                            for fileYearSingle in filesYears:
                                # check it is a .dat file
                                if fileYearSingle.endswith('.dat'):

                                    # keep 2019 for testing
                                    YearSingle = os.path.splitext(fileYearSingle)[0]
                                    if YearSingle == str(testYear):
                                        continue

                                    # display
                                    # print(fileYearSingle + ' ', end='')

                                    filePath = os.path.join(folderEachINVtype, fileYearSingle)

                                    # load
                                    file = open(filePath, 'rb')
                                    dataYear = pickle.load(file)
                                    file.close()

                                    # load features and target
                                    allDataTrainSingle = dataYear['allData']
                                    # targetSingle = dataYear['target']
                                    numFeaturesTrain = dataYear['numFeatures']
                                    # numSamples = dataYear['numSamples']
                                    # concat
                                    frames = [allDataTrain, allDataTrainSingle]
                                    allDataTrain = pd.concat(frames, axis=0)
                                    # frames = [targetTrain, targetSingle]
                                    # targetTrain = pd.concat(frames, axis=0)

                                    # print(' ', end='')

                    # remove negatives
                    posit = (allDataTrain['ac'] >= 0)
                    allDataTrain = allDataTrain.loc[posit]

                    # remove duplicates
                    allDataTrain.drop_duplicates(inplace=True)

                    util.print_pers('', fileResultNameFull)

                    # size of features
                    numSamplesTrain = allDataTrain.shape[0]

                    # ------------------------------------------------------
                    # select features and data loaders
                    # train
                    featuresTrain = allDataTrain.iloc[:, :numFeaturesTrain]
                    featuresTrain_tensor = torch.tensor(featuresTrain.values.astype(np.float32))
                    targetTrain_tensor = torch.tensor(allDataTrain['ac'].values.astype(np.float32))
                    targetTrain_tensor = torch.reshape(targetTrain_tensor, (numSamplesTrain, 1))
                    data_train = torch.utils.data.TensorDataset(featuresTrain_tensor, targetTrain_tensor)
                    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                                        batch_size=batch_sizeP_norm, shuffle=False,
                                                                        num_workers=numWorkersP, pin_memory=True,
                                                                        drop_last=True)

                    # select features
                    # test
                    featuresTest = allDataTest.iloc[:, :numFeaturesTest]
                    targetTest = allDataTest['ac']

                    # select one month for fine-tune train
                    for month_fineTuneTrain in range(0, 12):
                    # for week_fineTuneTrain in range(0, 52):

                        # print(month)
                        # weekStr = str(week_fineTuneTrain+1).rjust(2, '0')
                        # d = testYear.__str__() + '-W' + weekStr
                        # date_start = datetime.strptime(d + '-1', "%Y-W%W-%w")
                        # date_end = date_start + timedelta(days=6, hours=23)

                        monthStr = str(month_fineTuneTrain+1).rjust(2, '0')
                        lastDayMonth = calendar.monthrange(testYear, month_fineTuneTrain+1)[1]
                        date_start = datetime(int(testYear), month_fineTuneTrain+1, 1, 0, 0, 0)
                        date_end = datetime(int(testYear), month_fineTuneTrain+1, lastDayMonth, 23, 0, 0)

                        # features
                        featuresTest_fineTuneTrain = \
                            featuresTest[(featuresTest.index >= pd.to_datetime(date_start, utc=True)) &
                                         (featuresTest.index <= pd.to_datetime(date_end, utc=True))]
                        targetTest_fineTuneTrain = \
                            targetTest[(targetTest.index >= pd.to_datetime(date_start, utc=True)) &
                                         (targetTest.index <= pd.to_datetime(date_end, utc=True))]
                        # the first month we find data, we stop
                        if featuresTest_fineTuneTrain.shape[0] > 0:
                            break
                            # pass
                        else:
                            del featuresTest_fineTuneTrain

                    # size of features
                    numSamplesFineTune_train = featuresTest_fineTuneTrain.shape[0]

                    # use remaining months for testing/fine-tune testing
                    month_fineTuneTest = month_fineTuneTrain + 1
                    # week_fineTuneTest = week_fineTuneTrain + 1

                    # if no remaining months, we need to skip testYear
                    if (month_fineTuneTest+1) > 12:
                        # util.print_pers("", fileResultNameFull)
                        continue

                    monthStr = str(month_fineTuneTest+1).rjust(2, '0')
                    date_start = datetime(int(testYear), month_fineTuneTest+1, 1, 0, 0, 0)
                    date_end = datetime(int(testYear), 12, 31, 23, 0, 0)  # end of the year

                    # weekStr = str(week_fineTuneTest+1).rjust(2, '0')
                    # d = testYear.__str__() + '-W' + weekStr
                    # date_start = datetime.strptime(d + '-1', "%Y-W%W-%w")
                    # date_end = datetime(int(testYear), 12, 31, 23, 0, 0)  # end of the year

                    featuresTest_fineTuneTest = \
                        featuresTest[(featuresTest.index >= pd.to_datetime(date_start, utc=True)) &
                                     (featuresTest.index <= pd.to_datetime(date_end, utc=True))]
                    targetTest_fineTuneTest = \
                        targetTest[(targetTest.index >= pd.to_datetime(date_start, utc=True)) &
                                     (targetTest.index <= pd.to_datetime(date_end, utc=True))]
                    # if no data, we need to skip testYear
                    if featuresTest_fineTuneTest.shape[0] == 0:
                        # util.print_pers("", fileResultNameFull)
                        continue

                    # size of features
                    numSamplesTest = featuresTest_fineTuneTest.shape[0]

                    # select features and data loaders
                    # test fine-tune train
                    featuresTest_fineTuneTrain_tensor = torch.tensor(featuresTest_fineTuneTrain.values.astype(np.float32))
                    targetTest_fineTuneTrain_tensor = torch.tensor(targetTest_fineTuneTrain.values.astype(np.float32))
                    targetTest_fineTuneTrain_tensor = torch.reshape(targetTest_fineTuneTrain_tensor, (numSamplesFineTune_train, 1))
                    data_fineTuneTrain = torch.utils.data.TensorDataset(featuresTest_fineTuneTrain_tensor, targetTest_fineTuneTrain_tensor)
                    data_fineTuneTrain_loader = torch.utils.data.DataLoader(data_fineTuneTrain,
                                                                           batch_size=batch_sizeP, shuffle=False,
                                                                           num_workers=numWorkersP, pin_memory=True,
                                                                           drop_last=True)
                    # select features and data loaders
                    # test fine-tune test
                    featuresTest_fineTuneTest_tensor = torch.tensor(featuresTest_fineTuneTest.values.astype(np.float32))
                    targetTest_fineTuneTest_tensor = torch.tensor(targetTest_fineTuneTest.values.astype(np.float32))
                    targetTest_fineTuneTest_tensor = torch.reshape(targetTest_fineTuneTest_tensor, (numSamplesTest, 1))
                    data_fineTuneTest = torch.utils.data.TensorDataset(featuresTest_fineTuneTest_tensor, targetTest_fineTuneTest_tensor)
                    data_fineTuneTest_loader = torch.utils.data.DataLoader(data_fineTuneTest,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True)

                    if True:
                        # normalization
                        # util.print_pers('', fileResultNameFull)
                        util.print_pers('Normalization...', fileResultNameFull)
                        # train
                        # meanNormTrainFeatures, stdNormTrainFeatures, meanNormTrainTarget, stdNormTrainTarget = \
                            # util.computeMeanStd(data_train_loader, numSamplesTrain, batch_sizeP, cuda)
                        minNormTrainFeatures, maxNormTrainFeatures, minNormTrainTarget, maxNormTrainTarget = \
                            util.computeMinMax(data_train_loader, numSamplesTrain, batch_sizeP, cuda)

                        # print(minNormTrainFeatures)
                        # print(maxNormTrainFeatures)
                        # print(minNormTrainTarget)
                        # print(maxNormTrainTarget)
                        print()

                        # train
                        # apply normalization
                        # featuresTrain_tensor = util.applyZScoreNorm(featuresTrain_tensor, meanNormTrainFeatures, stdNormTrainFeatures)
                        # targetTrain_tensor = util.applyZScoreNorm(targetTrain_tensor, meanNormTrainTarget, stdNormTrainTarget)
                        featuresTrain_tensor = util.applyMinMaxNorm(featuresTrain_tensor, minNormTrainFeatures, maxNormTrainFeatures)
                        targetTrain_tensor = util.applyMinMaxNorm(targetTrain_tensor, minNormTrainTarget, maxNormTrainTarget)
                        # update data loader
                        data_train = torch.utils.data.TensorDataset(featuresTrain_tensor, targetTrain_tensor)
                        data_train_loader = torch.utils.data.DataLoader(data_train,
                                                                        batch_size=batch_sizeP, shuffle=False,
                                                                        num_workers=numWorkersP, pin_memory=True,
                                                                        drop_last=True)
                        # fine-tune train
                        # apply normalization
                        # featuresTest_tensor = util.applyZScoreNorm(featuresTest_tensor, meanNormTrainFeatures, stdNormTrainFeatures)
                        # targetTest_tensor = util.applyZScoreNorm(targetTest_tensor, meanNormTrainTarget, stdNormTrainTarget)
                        featuresTest_fineTuneTrain_tensor = util.applyMinMaxNorm(featuresTest_fineTuneTrain_tensor, minNormTrainFeatures, maxNormTrainFeatures)
                        targetTest_fineTuneTrain_tensor = util.applyMinMaxNorm(targetTest_fineTuneTrain_tensor, minNormTrainTarget, maxNormTrainTarget)
                        # update data loader
                        data_finetune_train = torch.utils.data.TensorDataset(featuresTest_fineTuneTrain_tensor, targetTest_fineTuneTrain_tensor)
                        data_finetune_train_loader = torch.utils.data.DataLoader(data_finetune_train,
                                                                               batch_size=batch_sizeP, shuffle=False,
                                                                               num_workers=numWorkersP, pin_memory=True,
                                                                               drop_last=True)
                        # test = fine-tune test
                        # apply normalization
                        # featuresTest_tensor = util.applyZScoreNorm(featuresTest_tensor, meanNormTrainFeatures, stdNormTrainFeatures)
                        # targetTest_tensor = util.applyZScoreNorm(targetTest_tensor, meanNormTrainTarget, stdNormTrainTarget)
                        featuresTest_fineTuneTest_tensor = util.applyMinMaxNorm(featuresTest_fineTuneTest_tensor, minNormTrainFeatures, maxNormTrainFeatures)
                        targetTest_fineTuneTest_tensor = util.applyMinMaxNorm(targetTest_fineTuneTest_tensor, minNormTrainTarget, maxNormTrainTarget)
                        # update data loader
                        data_finetune_test = torch.utils.data.TensorDataset(featuresTest_fineTuneTest_tensor, targetTest_fineTuneTest_tensor)
                        data_finetune_test_loader = torch.utils.data.DataLoader(data_finetune_test,
                                                                        batch_size=batch_sizeP, shuffle=False,
                                                                        num_workers=numWorkersP, pin_memory=True)

                    # extract validation subset
                    numSamplesVal = math.floor(numSamplesTrain / 10)
                    numSamplesTrain = numSamplesTrain - numSamplesVal
                    data_train, data_val = torch.utils.data.random_split(data_train, [numSamplesTrain, numSamplesVal],
                                                                         generator=torch.Generator().manual_seed(42))
                    # update train loader
                    # data_train = torch.utils.data.Subset(data_train, data_train_idx)
                    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True,
                                                                    drop_last=True)
                    # create val loader
                    # data_val = torch.utils.data.Subset(data_train, data_val_idx)
                    data_val_loader = torch.utils.data.DataLoader(data_val,
                                                                  batch_size=batch_sizeP, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)


                    # ------------------------------------------------------
                    # TRAIN MODEL

                    # display
                    util.print_pers('Training', fileResultNameFull)

                    # display
                    util.print_pers('\tNum samples train: {0}'.format(numSamplesTrain), fileResultNameFull)
                    util.print_pers('\tNum features train: {0}'.format(numFeaturesTrain), fileResultNameFull)
                    # util.print_pers('', fileResultNameFull)

                    # model
                    currentModel = model_1(numFeatures=numFeaturesTrain, numHidden=numHidden)
                    # currentModel = AngeloNet(numClasses=1)
                    if cuda:
                        currentModel.to('cuda')
                    # we want to train it
                    for param in currentModel.parameters():
                        param.requires_grad = True
                    # log
                    if log:
                        print()
                        print(currentModel)

                    # optim
                    optimizer_ft = optim.SGD(currentModel.parameters(), lr=2e-4, momentum=0.9)
                    # optimizer_ft = optim.LBFGS(currentModel.parameters(), max_iter=10000, lr=0.1)
                    # optimizer_ft = optim.Adam(currentModel.parameters(), lr=0.002)
                    criterion = nn.MSELoss()

                    # learning
                    dataset_sizes = {'train': numSamplesTrain,
                                     'val': numSamplesVal,
                                     'test': numSamplesTest}

                    currentModel = util.train_model(currentModel, criterion, optimizer_ft,
                                                    num_epochs_train, dataset_sizes,
                                                    data_train_loader, data_val_loader,
                                                    'modelGeno', fileResultNameFull, cuda,
                                                    batch_sizeP, minNormTrainTarget, maxNormTrainTarget)

                    # display
                    util.print_pers('', fileResultNameFull)


                    # ------------------------------------------------------
                    # TEST (model NO fine tune)

                    # display
                    util.print_pers('Testing (model NO fine tune)', fileResultNameFull)

                    # display
                    util.print_pers('\tNum samples test: {0}'.format(numSamplesTest), fileResultNameFull)
                    util.print_pers('\tNum features test: {0}'.format(numFeaturesTest), fileResultNameFull)

                    currentModel.eval()   # Set model to evaluate mode
                    # zero the parameter gradients
                    optimizer_ft.zero_grad()
                    torch.no_grad()

                    # loop on samples
                    # init
                    running_error = 0.0
                    outputALL_test = torch.zeros(numSamplesTest, 1)
                    labelsALL_test = torch.zeros(numSamplesTest, 1)
                    for batch_num, (inputs, labels) in enumerate(data_finetune_test_loader):

                        # get size of current batch
                        sizeCurrentBatch = labels.size(0)

                        # stack
                        indStart = batch_num * batch_sizeP
                        indEnd = indStart + sizeCurrentBatch

                        # extract features
                        if cuda:
                            inputs = inputs.to('cuda')
                            labels = labels.to('cuda')

                        # predict
                        with torch.set_grad_enabled(False):
                            outputs = currentModel(inputs)
                            if cuda:
                                outputs = outputs.to('cuda')

                            running_error += err_mae(util.reverseMinMax(outputs, minNormTrainTarget, maxNormTrainTarget),
                                                     util.reverseMinMax(labels, minNormTrainTarget, maxNormTrainTarget))

                            outputALL_test[indStart:indEnd, :] = outputs.cpu()
                            labelsALL_test[indStart:indEnd, :] = labels.cpu()

                    # accuracy
                    # errorResult = err_mae(util.reverseMinMax(outputALL_finetune_test, minNormTrainTarget, maxNormTrainTarget),
                    # util.reverseMinMax(labelsALL_finetune_test, minNormTrainTarget, maxNormTrainTarget))
                    # errorResult = errorResult / dataset_sizes_finetune['test']
                    errorResult = running_error.double() / dataset_sizes['test']

                    # print(output_test)
                    util.print_pers('\tTest error (MAE): {:.4f}'.format(errorResult), fileResultNameFull)

                    if True:
                        # dsBatch = 100
                        # dsBatch = len(outputALL_train)

                        outputALL_test_rev = util.reverseMinMax(outputALL_test, minNormTrainTarget, maxNormTrainTarget)
                        labelsALL_test_rev = util.reverseMinMax(labelsALL_test, minNormTrainTarget, maxNormTrainTarget)

                        lenBatch_test = len(outputALL_test_rev)
                        if dsBatch > lenBatch_test:
                            dsBatch = lenBatch_test
                        # index_ds_test = range(0, lenBatch_test, math.floor(lenBatch_test / dsBatch))
                        index_ds_test = range(0, dsBatch, 1)
                        outputALL_test_rev = outputALL_test_rev[index_ds_test]
                        labelsALL_test_rev = labelsALL_test_rev[index_ds_test]

                        """
                        fig = plt.figure()
                        plt.plot(outputALL_test_rev, 'r')
                        plt.plot(labelsALL_test_rev, 'g')
                        plt.title('Testing (model NO fine tune)')
                        plt.show()
                        fig.savefig('Test year {0} - Test module {1} - Testing (model NO fine tune).png'.format(testYear, folderPVcells[indexTest]))
                        """

                        print('', end='')

                    # assign
                    errorALL.append(errorResult.cpu())

                    util.print_pers('', fileResultNameFull)


                    # ------------------------------------------------------
                    # FINE TUNE MODEL

                    # display
                    util.print_pers('Fine tune', fileResultNameFull)

                    # split test db
                    numSamplesFineTune_train = len(data_finetune_train)
                    numSamplesFineTune_val = math.floor(numSamplesFineTune_train / 10)
                    numSamplesFineTune_train = numSamplesFineTune_train - numSamplesFineTune_val
                    numSamplesFineTune_test = numSamplesTest
                    data_finetune_train, data_finetune_val = \
                        torch.utils.data.random_split(data_finetune_train,
                                                      [numSamplesFineTune_train, numSamplesFineTune_val],
                                                                         generator=torch.Generator().manual_seed(42))
                    # fine tune loader - train
                    data_finetune_train_loader = torch.utils.data.DataLoader(data_finetune_train,
                                                                    batch_size=batch_sizeP, shuffle=False,
                                                                    num_workers=numWorkersP, pin_memory=True,
                                                                    drop_last=True)
                    # fine tune loader - val
                    data_finetune_val_loader = torch.utils.data.DataLoader(data_finetune_val,
                                                                  batch_size=batch_sizeP, shuffle=False,
                                                                  num_workers=numWorkersP, pin_memory=True)

                    # display
                    util.print_pers('\tNum samples fine-tune train: {0}'.format(numSamplesFineTune_train), fileResultNameFull)
                    util.print_pers('\tNum features fine-tune train: {0}'.format(numFeaturesTrain), fileResultNameFull)
                    util.print_pers('', fileResultNameFull)

                    # learning
                    dataset_sizes_finetune = {'train': numSamplesFineTune_train,
                                              'val': numSamplesFineTune_val,
                                              'test': numSamplesFineTune_test}

                    currentModel = util.train_model(currentModel, criterion, optimizer_ft,
                                                    num_epochs_finetune, dataset_sizes_finetune,
                                                    data_finetune_train_loader, data_finetune_val_loader,
                                                    'modelGeno', fileResultNameFull, cuda,
                                                    batch_sizeP, minNormTrainTarget, maxNormTrainTarget)

                    # inutile
                    print('', end='')


                    # ------------------------------------------------------
                    # TEST (model WITH fine tune)

                    # display
                    util.print_pers('Testing (model WITH fine tune)', fileResultNameFull)

                    # display
                    util.print_pers('\tNum samples fine-tune test: {0}'.format(numSamplesFineTune_test), fileResultNameFull)
                    util.print_pers('\tNum features fine-tune test: {0}'.format(numFeaturesTest), fileResultNameFull)

                    currentModel.eval()   # Set model to evaluate mode
                    # zero the parameter gradients
                    optimizer_ft.zero_grad()
                    torch.no_grad()

                    # loop on samples
                    # init
                    running_error = 0.0
                    outputALL_finetune_test = torch.zeros(numSamplesFineTune_test, 1)
                    labelsALL_finetune_test = torch.zeros(numSamplesFineTune_test, 1)
                    for batch_num, (inputs, labels) in enumerate(data_finetune_test_loader):

                        # get size of current batch
                        sizeCurrentBatch = labels.size(0)

                        # stack
                        indStart = batch_num * batch_sizeP
                        indEnd = indStart + sizeCurrentBatch

                        # extract features
                        if cuda:
                            inputs = inputs.to('cuda')
                            labels = labels.to('cuda')

                        # predict
                        with torch.set_grad_enabled(False):
                            outputs = currentModel(inputs)
                            if cuda:
                                outputs = outputs.to('cuda')

                            running_error += err_mae(util.reverseMinMax(outputs, minNormTrainTarget, maxNormTrainTarget),
                                             util.reverseMinMax(labels, minNormTrainTarget, maxNormTrainTarget))

                            outputALL_finetune_test[indStart:indEnd, :] = outputs.cpu()
                            labelsALL_finetune_test[indStart:indEnd, :] = labels.cpu()

                    # accuracy
                    # errorResult = err_mae(util.reverseMinMax(outputALL_finetune_test, minNormTrainTarget, maxNormTrainTarget),
                                          # util.reverseMinMax(labelsALL_finetune_test, minNormTrainTarget, maxNormTrainTarget))
                    # errorResult = errorResult / dataset_sizes_finetune['test']
                    errorResult = running_error.double() / dataset_sizes_finetune['test']

                    # print(output_test)
                    util.print_pers('\tTest error (MAE): {:.4f}'.format(errorResult), fileResultNameFull)

                    if True:
                        # dsBatch = 100
                        # dsBatch = len(outputALL_train)
                        # titleStr = 'Test module {0} - Test year {1} - \nTesting (model WITH fine tune).png'.format(folderPVcells[indexTest], testYear)
                        saveStr = '{0} - Test module {1} - Test year {2}'.format(city, indexTest+1, testYear)
                        titleStr = '{0}'.format(city)
                        fileNamePlot = os.path.join(folderResults, saveStr.replace('\n', '')) + '.png'

                        outputALL_finetune_test_rev = util.reverseMinMax(outputALL_finetune_test, minNormTrainTarget, maxNormTrainTarget)
                        labelsALL_finetune_test_rev = util.reverseMinMax(labelsALL_finetune_test, minNormTrainTarget, maxNormTrainTarget)

                        lenBatch_test = len(outputALL_finetune_test_rev)
                        # index_ds_test = range(0, lenBatch_test, math.floor(lenBatch_test / dsBatch))
                        index_ds_test = range(0, dsBatch, 1)
                        outputALL_finetune_test_rev = outputALL_finetune_test_rev[index_ds_test]
                        labelsALL_finetune_test_rev = labelsALL_finetune_test_rev[index_ds_test]
                        index_test = featuresTest_fineTuneTest.index[index_ds_test]
                        fig, axes = plt.subplots(1,1)
                        # plt.plot(outputALL_test_rev, 'r', label="Output (Digital Twin - without calibration)")
                        plt.plot(index_test, outputALL_test_rev, 'r', label="Output (Digital Twin - without calibration)")
                        plt.plot(index_test, outputALL_finetune_test_rev, 'b', label="Output (Digital Twin - with calibration)")
                        plt.plot(index_test, labelsALL_finetune_test_rev, 'g', label="Ground Truth")
                        plt.legend(loc="upper left")
                        # plt.xlabel('Time')
                        plt.ylabel('Energy output (Wh)')
                        # plt.title('Testing (model WITH fine tune)')
                        plt.title(titleStr)
                        axes.xaxis.set_major_locator(MaxNLocator(5))
                        # fig.locator_params(axis='x', nbins=4)
                        plt.show()
                        fig.savefig(fileNamePlot)

                        print('', end='')

                    # delete
                    del currentModel

                    # assign
                    errorALL_finetune.append(errorResult.cpu())

                    util.print_pers('', fileResultNameFull)

                # increase iteration
                i = i + 1

            # average accuracy over iterations and test years
            meanError = np.mean(errorALL)
            meanError_finetune = np.mean(errorALL_finetune)

            # delete
            # del errorALL, errorALL_finetune

            # display
            util.print_pers('Mean error (MAE) over {0} iterations: {1:.4f} (NO fine tune)'.format(len(errorALL),
                                                                                                  meanError),
                            fileResultNameFull)
            util.print_pers('Mean error (MAE) over {0} iterations: {1:.4f} (WITH fine tune)'.format(len(errorALL_finetune),
                                                                                                    meanError_finetune),
                            fileResultNameFull)
            util.print_pers('', fileResultNameFull)

    #close
    fileResult.close()

    # del
    # torch.no_grad()
    try:
        del currentModel
        del data_train, data_train_loader
        del data_finetune_train, data_finetune_train_loader
        del data_finetune_val, data_finetune_val_loader
        del data_finetune_test, data_finetune_test_loader
        del inputs, labels
        torch.cuda.empty_cache()
    except:
        print('')

    print()


