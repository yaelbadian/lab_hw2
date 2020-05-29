import preprocessing
import model
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor


def model_pipeline(train_dataset, test_dataset, batch_size, num_epochs, optimizer, dropout=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    net = model.MaskDetector(train_dataset.df)
    return model.fit(net, train_loader, test_loader, num_epochs, optimizer, plot=True, save=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    parser.add_argument('output_folder', type=str, help='Output folder path, containing images')

    args = parser.parse_args()
    train_path = args.input_folder
    test_path = args.output_folder

    transformations = Compose([
        ToPILImage('RGB'),
        Resize((100, 100)),
        ToTensor(),  # [0, 1]
    ])

    train_dataset = preprocessing.FaceMaskDataset(train_path, transformations)
    test_dataset = preprocessing.FaceMaskDataset(test_path, transformations)
    model_pipeline(train_dataset, test_dataset, 32, 2, 'Adam')


    # train_iter = iter(data_loader)
    # images, labels = train_iter.next()
    # preprocessing.show_images(images, labels)