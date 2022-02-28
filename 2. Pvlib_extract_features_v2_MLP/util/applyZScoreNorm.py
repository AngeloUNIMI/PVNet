def applyZScoreNorm(inputTensor, mean, std):
    x = inputTensor - mean
    x = x / std
    return x
