import numpy as np
import torch

def computeMinMax(dataloader, dataset_sizes, batch_sizeP, cuda):

    numBatches = np.round(dataset_sizes / batch_sizeP)

    # ---------------------------------------------
    # norm for data
    pop_minData = []
    pop_maxData = []
    for i, (data, y) in enumerate(dataloader):

        if cuda:
            data = data.to('cuda')

        # shape (3,)
        batch_min = torch.min(data)
        batch_max = torch.max(data)

        if cuda:
            batch_min = batch_min.detach().to('cpu')
            batch_max = batch_max.detach().to('cpu')

        pop_minData.append(batch_min)
        pop_maxData.append(batch_max)


    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_minData = np.amin(pop_minData)
    pop_maxData = np.amax(pop_maxData)

    # ---------------------------------------------
    # norm for target
    pop_minTarget = []
    pop_maxTarget = []
    for i, (data, y) in enumerate(dataloader):

        if cuda:
            y = y.to('cuda')

        # shape (3,)
        batch_min = torch.min(y)
        batch_max = torch.max(y)

        if cuda:
            batch_min = batch_min.detach().to('cpu')
            batch_max = batch_max.detach().to('cpu')

        pop_minTarget.append(batch_min)
        pop_maxTarget.append(batch_max)


    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_minTarget = np.amin(pop_minTarget)
    pop_maxTarget = np.amax(pop_maxTarget)

    # ---------------------------------------------
    return pop_minData, pop_maxData, pop_minTarget, pop_maxTarget
