def reverseMinMax(inputTensor, min, max):
    x = inputTensor * (max-min)
    x = x + min
    return x
