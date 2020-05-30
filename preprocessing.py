import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from os import listdir
from os.path import isfile, join
import cv2
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
import matplotlib.pyplot as plt


# def show_images(images, labels):
#     grid = torchvision.utils.make_grid(images)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))
#     plt.axis('off')
#     plt.title(list(labels.numpy()))
#     plt.savefig('fig.jpg', format='jpg')
#     plt.close()


class FaceMaskDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        transormations = [ToPILImage('RGB'), Resize((100, 100)), ToTensor()]
        self.folder_path = folder_path
        self.df = self.create_data_df(folder_path)
        self.transform = Compose(transormations[:-1] + transform + [transormations[-1]])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, id):
        if isinstance(id, slice):
            raise NotImplementedError('slicing is not supported')
        row = self.df.iloc[id]
        image_id = row['id']
        image = cv2.imread(join(self.folder_path, image_id))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        label = row['label']
        return image, label, image_id

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


def show_images(dataset, images_ids, file_name):
    images, labels = [], []
    wanted_images = dataset.df[dataset.df['id'].isin(images_ids)]
    for _, row in wanted_images.iterrows():
        image = cv2.imread(join(dataset.folder_path, row['id']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = dataset.transform(image)
        label = row['label']
        images.append(image)
        labels.append(label)
    grid = torchvision.utils.make_grid(images)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.title(labels)
    plt.savefig('plots/' + file_name + '.jpg', format='jpg')
    plt.close()





