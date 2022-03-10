import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torchvision.ops.deform_conv import DeformConv2d

class DeformCNN(nn.Module):
    def __init__(self):
        super(DeformCNN, self).__init__()
        self.offset1 = nn.Conv2d(in_channels=1,
                                 out_channels=2 * 5 *5,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)
        self.DeformConv1 = DeformConv2d(in_channels=1,
                                        out_channels=16,
                                        kernel_size=5,
                                        stride=1,
                                        padding=2)
        self.pool1 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.offset2 = nn.Conv2d(in_channels=16,
                                 out_channels=2 * 5 *5,
                                 kernel_size=5,
                                 stride=1,
                                 padding=2)
        self.DeformConv2 = DeformConv2d(16, 32, 5, 1, 2)
        self.pool2 = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        offset = self.offset1(x)
        x = self.DeformConv1(x, offset)
        x = self.pool1(x)
        offset = self.offset2(x)
        x = self.DeformConv2(x, offset)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)


        # x = self.conv1(x)
        # x = self.conv2(x)
        # # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        # x = x.view(x.size(0), -1)
        # output = self.out(x)
        return output

class DeformCNN_Module(pl.LightningModule):
    def __init__(self, lr):
        super(DeformCNN_Module, self).__init__()
        self.lr = lr
        self.net = DeformCNN()
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



if __name__ == '__main__':
    net = DeformCNN()
    x = torch.rand([4,1,28,28])

    offset = net.offset1(x)
    x = net.DeformConv1(x, offset)
    print(x.shape) # 4, 16, 28, 28


    x = net.pool1(x)
    print(x.shape) # 4, 16, 14, 14

    offset = net.offset2(x)
    x = net.DeformConv2(x, offset)
    print(x.shape) # 4, 32, 14, 14

    x = net.pool2(x)
    print(x.shape) # [4, 32, 7, 7]

    x = x.view(x.size(0), -1)
    print(x.shape) # [4, 1568]

    output = net.out(x)
    print(output.shape) # 4, 10

