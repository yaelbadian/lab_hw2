import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.init as init
from torch.nn import Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU, Sequential, Softmax, Module, BatchNorm2d, Dropout
from train import to_gpu

class MaskDetector(Module):
    def __init__(self, train_df):
        super(MaskDetector, self).__init__()
        self.train_df = train_df

        self.convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            # BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            # BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            # BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer4 = Sequential(
            Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            # BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )


        self.linearLayers = Sequential(
            Linear(in_features=2304, out_features=512),
            Dropout(p=0.1),
            ReLU(),
            Linear(in_features=512, out_features=2),
            Dropout(p=0.1),
            Softmax(dim=1)
        )

        # Initialize layers' weights
        for sequential in [self.convLayer1, self.convLayer2, self.convLayer3, self.convLayer4, self.linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = self.convLayer4(out)
        out = out.view(-1, 2304)
        out = self.linearLayers(out)
        return out
    def configure_weighted_loss(self):
        balanced = self.train_df['label'].mean()
        weights = torch.tensor([1-balanced, balanced])
        weights = to_gpu(weights)
        return CrossEntropyLoss(weight=weights)

    def configure_optimizer(self, opt_str):
        if opt_str == 'Adam':
            opt = torch.optim.Adam(self.parameters())
        elif opt_str == 'SGD':
            opt = torch.optim.SGD(self.parameters(), lr=0.005, momentum=0.9)
        elif opt_str == 'Adadelta':
            opt = torch.optim.Adadelta(self.parameters())
        elif opt_str == 'RMSprop':
            opt = torch.optim.RMSprop(self.parameters())
        else:
            opt = torch.optim.Adam(self.parameters())
        return opt

    def visualize_conv2d_features(self, conv_name, file_name):
        conv = getattr(self, conv_name)
        tensor = conv[0].weight.data.detach().cpu()
        n, c, w, h = tensor.shape
        tensor = tensor.view(n * c, -1, w, h)
        rows = np.min((tensor.shape[0] // 8 + 1, 64))
        grid = torchvision.utils.make_grid(tensor, nrow=8, normalize=True, padding=1)
        plt.figure(figsize=(8, rows))
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.savefig('plots/' + file_name + '.jpg', format='jpg')
        plt.close()