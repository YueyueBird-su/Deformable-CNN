import pytorch_lightning as pl
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

class MNIST_DataModule(pl.LightningDataModule):
    def __init__(self, batchsize, numworkers):
        super(MNIST_DataModule, self).__init__()
        self.batchsize = batchsize
        self.numworkers = numworkers


    def setup(self, stage = None) -> None:
        self.train_data = MNIST(
            root='data',
            train=True,
            transform=transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()]),
            download=True,
        )
        self.val_data = MNIST(
            root='data',
            train=False,
            transform=transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()]),
        )

    def train_dataloader(self) :
        return DataLoader(self.train_data, batch_size=self.batchsize, shuffle= True, num_workers=self.numworkers, )

    def val_dataloader(self) :
        return DataLoader(self.val_data, batch_size=self.batchsize, shuffle= False, num_workers=self.numworkers, )

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batchsize, shuffle=False, num_workers=self.numworkers, )