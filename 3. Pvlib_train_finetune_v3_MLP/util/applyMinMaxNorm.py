def applyMinMaxNorm(inputTensor, min, max):
    x = inputTensor - min
    x = x / (max-min)
    return x
