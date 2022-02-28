import numpy as np
import torch

def computeMeanStd(dataloader, dataset_sizes, batch_sizeP, cuda):

    numBatches = np.round(dataset_sizes / batch_sizeP)

    # ---------------------------------------------
    # norm for data
    pop_meanData = []
    pop_std0Data = []
    for i, (data, y) in enumerate(dataloader):

        if cuda:
            data = data.to('cuda')

        # shape (3,)
        batch_mean = torch.mean(data)
        batch_std0 = torch.std(data)

        if cuda:
            batch_mean = batch_mean.detach().to('cpu')
            batch_std0 = batch_std0.detach().to('cpu')

        pop_meanData.append(batch_mean)
        pop_std0Data.append(batch_std0)


    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_meanData = np.mean(pop_meanData)
    pop_std0Data = np.mean(pop_std0Data)

    # ---------------------------------------------
    # norm for target
    pop_meanTarget = []
    pop_std0Target = []
    for i, (data, y) in enumerate(dataloader):

        if cuda:
            y = y.to('cuda')

        # shape (3,)
        batch_mean = torch.mean(y)
        batch_std0 = torch.std(y)

        if cuda:
            batch_mean = batch_mean.detach().to('cpu')
            batch_std0 = batch_std0.detach().to('cpu')

        pop_meanTarget.append(batch_mean)
        pop_std0Target.append(batch_std0)


    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_meanTarget = np.mean(pop_meanTarget)
    pop_std0Target = np.mean(pop_std0Target)

    # ---------------------------------------------
    return pop_meanData, pop_std0Data, pop_meanTarget, pop_std0Target
