import preprocessing
import model
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, ColorJitter, RandomHorizontalFlip
from pytorch_model_summary import summary
import torch


def model_pipeline(train_dataset, test_dataset, batch_size, num_epochs, optimizer, dropout=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    net = model.MaskDetector(train_dataset.df)
    print(summary(net, torch.zeros((1, 3, 100, 100)), show_input=False, show_hierarchical=True))

    model_net, best_test_f1 = model.fit(net, train_loader, test_loader, num_epochs, optimizer, plot=True, save=True)
    net.visualize_conv2d_features('convLayer1', 'convLayer1')
    net.visualize_conv2d_features('convLayer2', 'convLayer2')
    net.visualize_conv2d_features('convLayer3', 'convLayer3')
    return model_net, best_test_f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    parser.add_argument('output_folder', type=str, help='Output folder path, containing images')

    args = parser.parse_args()
    train_path = args.input_folder
    test_path = args.output_folder

    transformations = [RandomCrop(90, pad_if_needed=True, padding_mode='edge'),
                       ColorJitter(0.125, 0.125, 0.125, 0.1),
                       RandomHorizontalFlip()]

    train_dataset = preprocessing.FaceMaskDataset(train_path, transformations)
    test_dataset = preprocessing.FaceMaskDataset(test_path, [])
    model_pipeline(train_dataset, test_dataset, 256, 100, 'Adam')


    # train_iter = iter(data_loader)
    # images, labels = train_iter.next()
    # preprocessing.show_images(images, labels)