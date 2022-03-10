from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from CNN_Module import CNN_Module
from DeformCNN_Module import DeformCNN_Module
from MNIST_DataModule import MNIST_DataModule

from pytorch_lightning.callbacks import EarlyStopping

cnn_model = CNN_Module(lr=0.001)
deform_model = DeformCNN_Module(lr=0.001)

dataModule = MNIST_DataModule(batchsize=16,
                              numworkers=16)

logger = TensorBoardLogger(save_dir='./tb_train_log',)


trainer = Trainer(gpus=1,
                  logger = logger,
                  max_epochs=5)

# trainer.fit(cnn_model, dataModule)
# trainer.fit(deform_model, dataModule)

# trainer.test(cnn_model, dataModule, ckpt_path="/home/lpengsu/Study21/22_Study_Deformable-ConvNets/Deformable_Conv/MNIST/tb_train_log/default/conv_train/checkpoints/epoch=4-step=18749.ckpt")
trainer.test(deform_model, dataModule, ckpt_path="/Deformable_Conv/MNIST/tb_train_log/default/deformConv_train/checkpoints/epoch=4-step=18749.ckpt")

