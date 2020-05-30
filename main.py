import preprocessing
import model
import argparse
import os
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop, ColorJitter, RandomHorizontalFlip
from pytorch_model_summary import summary
import torch


def model_pipeline(train_dataset, test_dataset, batch_size, num_epochs, optimizer, dropout=None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    net = model.MaskDetector(train_dataset.df)
    print(summary(net, torch.zeros((1, 3, 100, 100)), show_input=False, show_hierarchical=True))
    model_net = model.fit(net, train_loader, test_loader, num_epochs, optimizer, plot=True, save=True)
    net.visualize_conv2d_features('convLayer1', 'convLayer1')
    net.visualize_conv2d_features('convLayer2', 'convLayer2')
    net.visualize_conv2d_features('convLayer3', 'convLayer3')
    return model_net


if __name__ == '__main__':
    # parse parameters
    parser = argparse.ArgumentParser(description='Process input')
    parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
    parser.add_argument('output_folder', type=str, help='Output folder path, containing images')
    args = parser.parse_args()
    train_path = args.input_folder
    test_path = args.output_folder

    # create folders
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # set data configurations
    transformations = [RandomCrop(90, pad_if_needed=True, padding_mode='edge'),
                       ColorJitter(0.125, 0.125, 0.125, 0.1),
                       RandomHorizontalFlip()]
    train_dataset = preprocessing.FaceMaskDataset(train_path, transformations)
    test_dataset = preprocessing.FaceMaskDataset(test_path, [])

    # train the model
    model_net = model_pipeline(train_dataset, test_dataset, 256, 2, 'Adam')
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)
    predictions = model.predict(model_net, test_loader)
    f1, roc_auc = model.calculate_scores(predictions)
    print('F1 score:', f1, 'ROC AUC score:', roc_auc)
    mistakes = predictions[predictions['pred'] != predictions['true']]['id'].values[:10]
    preprocessing.show_images(test_dataset, mistakes, 'mistakes')
