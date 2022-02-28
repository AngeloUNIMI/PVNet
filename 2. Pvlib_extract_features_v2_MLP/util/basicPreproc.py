import matplotlib.pyplot as plt


def basicPreproc(allData):

    # replace nans with 0
    # features = featuresSingle.replace(np.nan, 0)
    # target = targetSingle.replace(np.nan, 0)
    # remove nans
    allData = allData.dropna(axis=0)

    """
    idx1 = features.index
    idx2 = targetSingle.index
    idx3 = idx1.intersection(idx2)
    target = targetSingle.loc[idx3]
    #
    target = target.dropna(axis=0)
    idx1 = features.index
    idx2 = target.index
    idx3 = idx1.intersection(idx2)
    features = features.loc[idx3]
    """

    """
    idx1 = target.index
    idx2 = features.index
    idx3 = idx1.intersection(idx2)
    print(len(idx1))
    print(len(idx2))
    print(len(idx3))
    print('---')
    print()
    """

    # plot labels
    if False:
        plt.figure()
        plt.plot(target['i_sc'], 'r')
        plt.show()

    return allData

    # target = targetSingle.dropna(axis=0)
    # remove negatives from targets
    # target_temp = target[(target['i_sc'] < 0)]
    # indexToRemove = target_temp.copy().index
    # target = target.drop(indexToRemove)
    # features = features.drop(indexToRemove)

    # util.print_pers('', fileResultNameFull)