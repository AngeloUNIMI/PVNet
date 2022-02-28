import torch
import torch.nn as nn

class model_1(nn.Module):
    def __init__(self, numFeatures, numHidden):
        super(model_1, self).__init__()

        self.classifierG = nn.Sequential(

            nn.Linear(numFeatures, numHidden),
            nn.BatchNorm1d(numHidden),
            nn.Sigmoid(),

            # nn.Linear(numHidden, numHidden),
            # nn.BatchNorm1d(numHidden),
            # nn.Sigmoid(),

            nn.Linear(numHidden, 1),

            # nn.ReLU(inplace=True),
            # nn.Tanh(),

        )

        # init
        for layer in self.classifierG:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    # Defining the forward pass
    def forward(self, x):
        x = self.classifierG(x)
        return x

class AngeloNet(nn.Module):
    def __init__(self, numClasses):
        super(AngeloNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Linear(50, numClasses)
        )

    def forward(self, x):

        x = x.view(-1, 1, 6, 3)

        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return x