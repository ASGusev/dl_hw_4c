import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size):
        super(DCGenerator, self).__init__()
        self.fc = nn.Linear(100, 16)
        self.deconvolution = nn.Sequential(
            nn.ConvTranspose2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, data):
        features = self.fc(data)
        picture = self.deconvolution(features.view(-1, 1, 4, 4))
        return picture


class DCDiscriminator(nn.Module):

    def __init__(self, image_size):
        super(DCDiscriminator, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, data):
        return self.convolutions(data)
