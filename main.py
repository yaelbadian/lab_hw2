import preprocessing
import model
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    args = parser.parse_args()
    folder_path = args.input_folder

    transformations = Compose([
        ToPILImage(),
        Resize((100, 100)),
        ToTensor(),  # [0, 1]
    ])

    dataset = preprocessing.FaceMaskDataset(folder_path, transformations)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    train_iter = iter(data_loader)
    images, labels = train_iter.next()
    dataset.show_images(images, labels)