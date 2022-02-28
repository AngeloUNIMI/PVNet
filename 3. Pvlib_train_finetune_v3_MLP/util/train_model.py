import torch
import time
import copy
from util.print_pers import print_pers
from util.reverseMinMax import reverseMinMax
import matplotlib.pyplot as plt
import math

torch.Tensor.ndim = property(lambda self: len(self.shape))


def err_mape(outputsF, labelsF):
    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    diff = torch.sub(labelsF, outputsF)
    scaled = torch.div(diff, labelsF)
    absol = torch.abs(scaled)
    MAPE = torch.sum(absol)

    # diff = torch.sub(labelsF, outputsF)
    # squared = torch.square(diff)
    # MSE = torch.sum(squared)

    return MAPE
    # return MSE

def err_mae(outputsF, labelsF):
    # https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    diff = torch.sub(labelsF, outputsF)
    absol = torch.abs(diff)
    MAE = torch.sum(absol)

    # diff = torch.sub(labelsF, outputsF)
    # squared = torch.square(diff)
    # MSE = torch.sum(squared)

    return MAE
    # return MSE


def train_model(model, criterion, optimizer,
                num_epochs, dataset_sizes,
                dataloader_train, dataloader_val,
                modelName, fileResultNameFull, cuda,
                batch_sizeP, minNormTrainTarget, maxNormTrainTarget):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    min_val_loss = 1e6
    best_error= 1e6

    for epoch in range(num_epochs):

        # init all data for plotting
        outputALL_train = torch.zeros(dataset_sizes['train'], 1)
        labelsALL_train = torch.zeros(dataset_sizes['train'], 1)
        outputALL_val = torch.zeros(dataset_sizes['val'], 1)
        labelsALL_val = torch.zeros(dataset_sizes['val'], 1)

        # display
        print_pers('\tEpoch {}/{}'.format(epoch+1, num_epochs), fileResultNameFull)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_error = 0.0

            # choose dataloader
            if phase == 'train':
                dataloaders_chosen = dataloader_train
            if phase == 'val':
                dataloaders_chosen = dataloader_val

            # Iterate over data.
            for batch_num, (inputs, labels) in enumerate(dataloaders_chosen):

                # get size of current batch
                sizeCurrentBatch = labels.size(0)

                # stack
                indStart = batch_num * batch_sizeP
                indEnd = indStart + sizeCurrentBatch

                if cuda:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        outputALL_train[indStart:indEnd, :] = outputs.cpu().detach()
                        labelsALL_train[indStart:indEnd, :] = labels.cpu().detach()

                    if phase == 'val':
                        outputALL_val[indStart:indEnd, :] = outputs.cpu()
                        labelsALL_val[indStart:indEnd, :] = labels.cpu()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_error += err_mae(reverseMinMax(outputs, minNormTrainTarget, maxNormTrainTarget),
                                             reverseMinMax(labels, minNormTrainTarget, maxNormTrainTarget))
                    # running_error += err_mae(outputs, labels)

            # compute epochs losses
            with torch.no_grad():
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_error = running_error.double() / dataset_sizes[phase]

            # display
            print_pers('\t\t{} Loss: {:.4f}; Error (MAE): {:.4f}'.format(phase, epoch_loss, epoch_error),
                       fileResultNameFull)

            # deep copy the model
            if (phase == 'val') and (epoch_error < best_error):
                best_error = epoch_error
                best_model_wts = copy.deepcopy(model.state_dict())
            if (phase == 'val') and (epoch_error == best_error) and (epoch_loss < min_val_loss):
                min_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        # plot outputs vs labels for the epoch (val)
        if False and phase == 'val':
            dsBatch = 100
            # dsBatch = len(outputALL_train)

            outputALL_train_rev = reverseMinMax(outputALL_train, minNormTrainTarget, maxNormTrainTarget)
            labelsALL_train_rev = reverseMinMax(labelsALL_train, minNormTrainTarget, maxNormTrainTarget)

            lenBatch_train = len(outputALL_train_rev)
            # index_ds_train = range(0, lenBatch_train, math.floor(lenBatch_train / dsBatch))
            index_ds_train = range(0, dsBatch, 1)
            outputALL_train_rev = outputALL_train_rev[index_ds_train]
            labelsALL_train_rev = labelsALL_train_rev[index_ds_train]
            plt.figure()
            plt.plot(outputALL_train_rev, 'r')
            plt.plot(labelsALL_train_rev, 'g')
            plt.title('Epoch: {}; Training'.format(epoch+1))
            plt.show()

            outputALL_val_rev = reverseMinMax(outputALL_val, minNormTrainTarget, maxNormTrainTarget)
            labelsALL_val_rev = reverseMinMax(labelsALL_val, minNormTrainTarget, maxNormTrainTarget)

            lenBatch_val = len(outputALL_val_rev)
            # index_ds_val = range(0, lenBatch_val, math.floor(lenBatch_val / dsBatch))
            if lenBatch_val < dsBatch:
                index_ds_val = range(0, lenBatch_val, 1)
            else:
                index_ds_val = range(0, dsBatch, 1)
            outputALL_val_rev = outputALL_val_rev[index_ds_val]
            labelsALL_val_rev = labelsALL_val_rev[index_ds_val]
            plt.figure()
            plt.plot(outputALL_val_rev, 'r')
            plt.plot(labelsALL_val_rev, 'g')
            plt.title('Epoch: {}; Validation'.format(epoch+1))
            plt.show()

            print('', end='')

    time_elapsed = time.time() - since
    print_pers('\tTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60),
               fileResultNameFull)
    print_pers('\tBest val error: {:4f}'.format(best_error), fileResultNameFull)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # save final
    # torch.save(model.state_dict(), os.path.join(dirResults, fileNameSaveFinal))

    print()

    # del
    torch.cuda.empty_cache()
    del inputs, labels
    del outputs, loss

    return model
