import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import cv2
from torch import long, tensor
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import matplotlib.pyplot as plt


def show_images(images, labels):
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(list(labels.numpy()))
    plt.savefig('fig.jpg', format='jpg')


class FaceMaskDataset(Dataset):

    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.df = self.create_data_df(folder_path)
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, id):
        if isinstance(id, slice):
            raise NotImplementedError('slicing is not supported')

        row = self.df.iloc[id]
        image = cv2.imread(join(self.folder_path, row['id']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        print('image {} shape:{}'.format(row['id'], image.shape))
        label = tensor([row['label']], dtype=long)
        return image, label

    @staticmethod
    def create_data_df(folder_path):
        df = []
        for file in listdir(folder_path):
            if isfile(join(folder_path, file)) and file.endswith('.jpg') and '_' in file:
                label = file.split('.jpg')[0].split('_')[1]
                if label.isnumeric():
                    label = int(label)
                df.append({'id': file, 'label': label})
        return pd.DataFrame(df)





