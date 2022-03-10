import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__() # 4, 1, 28, 28
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ), # 4, 16, 28, 28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 4, 16, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # 4, 32, 14, 14
            nn.ReLU(),
            nn.MaxPool2d(2), # 4, 32, 7, 7
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10) # 4, 10

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

class CNN_Module(pl.LightningModule):
    def __init__(self, lr):
        super(CNN_Module, self).__init__()
        self.lr = lr
        self.net = CNN()
        self.loss_func = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return optim.Adam(self.net.parameters(), lr=self.lr)

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_func(y_hat, y)
        acc = (torch.argmax(y_hat, 1) == y).sum().item() / float(y.shape[0])
        self.log("train_acc", acc, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_func(y_hat, y)
        self.log('Loss_val', loss, prog_bar=True, logger=True)
        acc = (torch.argmax(y_hat, 1) == y).sum().item() / float(y.shape[0])
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def test_step(self, batch, idx):
        x, y = batch
        y_hat = self.net(x)
        loss = self.loss_func(y_hat, y)
        self.log('Loss_test', loss, prog_bar=True, logger=True)
        acc = (torch.argmax(y_hat, 1) == y).sum().item() / float(y.shape[0])
        self.log("test_acc", acc, prog_bar=True, logger=True)
