import numpy as np
import pandas as pd
import datetime, copy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torchvision
import torch.nn.init as init
from torch.nn import Conv2d, CrossEntropyLoss, Linear, MaxPool2d, ReLU, Sequential, Softmax, Module, BatchNorm2d, Dropout


class MaskDetector(Module):
    def __init__(self, train_df):
        super(MaskDetector, self).__init__()
        self.train_df = train_df

        self.convLayer1 = Sequential(
            Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer2 = Sequential(
            Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.convLayer3 = Sequential(
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(3, 3)),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=(2, 2))
        )

        self.linearLayers = Sequential(
            Linear(in_features=2048, out_features=1024),
            Dropout(p=0.1),
            ReLU(),
            Linear(in_features=1024, out_features=2),
            Dropout(p=0.1),
            Softmax(dim=1)
        )

        # Initialize layers' weights
        for sequential in [self.convLayer1, self.convLayer2, self.convLayer3, self.linearLayers]:
            for layer in sequential.children():
                if isinstance(layer, (Linear, Conv2d)):
                    init.xavier_uniform_(layer.weight)

    def forward(self, x):
        out = self.convLayer1(x)
        out = self.convLayer2(out)
        out = self.convLayer3(out)
        out = out.view(-1, 2048)
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


def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


def plot_loss_and_error(train_f1s, train_roc_aucs, train_losses, test_f1s, test_roc_aucs, test_losses, model_name):
    fig = plt.figure(figsize=(12, 8))
    loss_fig = fig.add_subplot(3, 1, 1)
    loss_fig.plot(list(range(len(train_losses))), train_losses, label='Train Loss', color='turquoise', lw=2)
    loss_fig.plot(list(range(len(test_losses))), test_losses, label='Test Loss', color='orchid', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss as function of Epochs')
    error_fig = fig.add_subplot(3, 1, 2)
    error_fig.plot(list(range(len(train_f1s))), train_f1s, label='Train F1', color='turquoise', lw=2)
    error_fig.plot(list(range(len(test_f1s))), test_f1s, label='Test F1', color='orchid', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.title('F1 as function of Epochs')
    error_fig = fig.add_subplot(3, 1, 3)
    error_fig.plot(list(range(len(train_roc_aucs))), train_roc_aucs, label='Train Roc Auc', color='turquoise', lw=2)
    error_fig.plot(list(range(len(test_roc_aucs))), test_roc_aucs, label='Test Roc Auc', color='orchid', lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.legend()
    plt.title('ROC AUC as function of Epochs')
    plt.tight_layout()
    fig.savefig(f'plots/loss_plot_{model_name}.png')
    plt.close()



def predict(net, test_loader, criterion=None):
    current_test_losses = []
    net.eval()
    y_true, y_pred, y_scores, y_ids = np.array([]), np.array([]), np.array([]), np.array([])
    for images, labels, image_ids in test_loader:
        images = to_gpu(images)
        labels = to_gpu(labels)
        outputs = net(images)
        if criterion is not None:
            loss = criterion(outputs, labels)
            current_test_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        y_ids = np.concatenate([y_ids, np.array(image_ids)], axis=0)
        y_scores = np.concatenate([y_scores, outputs[:, 1].detach().cpu().numpy()], axis=0)
        y_true = np.concatenate([y_true, labels.detach().cpu().numpy()], axis=0)
        y_pred = np.concatenate([y_pred, predicted.detach().cpu().numpy()], axis=0)
    df = pd.DataFrame(zip(y_ids, y_scores, y_pred, y_true), columns=['id', 'score', 'pred', 'true'])
    df = df.astype({'pred':'int', 'true':'int'})
    net.train()
    if criterion is not None:
        test_loss = sum(current_test_losses) / len(current_test_losses)
        return df, test_loss
    else:
        return df


def calculate_scores(df):
    f1 = f1_score(df['true'], df['pred'])
    roc_auc = roc_auc_score(df['true'], df['score'])
    return f1, roc_auc


def fit(net, train_loader, test_loader, num_epochs=10, optimizer=None, plot=True, save=True):
    net = to_gpu(net)
    now = datetime.datetime.now()
    criterion = net.configure_weighted_loss()
    optimizer = net.configure_optimizer(optimizer)
    datetime_str = str(f'{now.date()}_{now.hour}:{now.minute}')
    print(datetime_str)
    train_f1s, train_roc_aucs, train_losses, test_f1s, test_roc_aucs, test_losses = [], [], [], [], [], []
    best_model_wts, best_test_f1, best_epoch = copy.deepcopy(net.state_dict()), 0.0, 0
    # training
    for epoch in range(num_epochs):
        # train
        current_train_losses = []
        for i, (images, labels, _) in enumerate(train_loader):
            images = to_gpu(images)
            labels = to_gpu(labels)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            current_train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        if plot or save:
            # train error & loss
            train_loss = sum(current_train_losses) / len(current_train_losses)
            train_losses.append(train_loss)
            df = predict(net, train_loader)
            train_f1, train_roc_auc = calculate_scores(df)
            train_f1s.append(train_f1)
            train_roc_aucs.append(train_roc_auc)
            # test error & loss
            df, test_loss = predict(net, test_loader, criterion)
            test_f1, test_roc_auc = calculate_scores(df)
            test_f1s.append(test_f1)
            test_roc_aucs.append(test_roc_auc)
            test_losses.append(test_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}] - Train: F1 {train_f1:.2f}| ROC_AUC {train_roc_auc:.2f}| '
                  f'Loss {train_loss:.2f} \t Test: F1:{test_f1:.2f}, ROC_AUC {test_roc_auc:.2f} | Loss {test_loss:.2f}')
            # updating the best model so far
            if test_f1 > best_test_f1:
                best_epoch = epoch
                best_test_f1 = test_f1
                model_name = f'model_{datetime_str}_{best_epoch}_{best_test_f1:.2f}'
                torch.save(net.state_dict(), 'models/' + model_name + '.pkl')
                print(f"Current Best Epoch: [{epoch}/{num_epochs}]\t Test F1: {best_test_f1:.2f}")
                plot_loss_and_error(train_f1s, train_roc_aucs, train_losses, test_f1s, test_roc_aucs, test_losses, model_name)
    # plotting
    if plot:
        plot_loss_and_error(train_f1s, train_roc_aucs, train_losses, test_f1s, test_roc_aucs, test_losses, 'last_plot')
    return net
