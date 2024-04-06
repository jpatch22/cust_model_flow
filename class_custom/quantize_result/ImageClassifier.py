# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class ImageClassifier(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.module_0 = py_nndct.nn.Input() #ImageClassifier::input_0(ImageClassifier::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ImageClassifier::ImageClassifier/Conv2d[conv1]/ret.3(ImageClassifier::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #ImageClassifier::ImageClassifier/ReLU[relu]/ret.5(ImageClassifier::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #ImageClassifier::ImageClassifier/MaxPool2d[pool]/464(ImageClassifier::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ImageClassifier::ImageClassifier/Conv2d[conv2]/ret.7(ImageClassifier::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #ImageClassifier::ImageClassifier/ReLU[relu]/ret.9(ImageClassifier::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #ImageClassifier::ImageClassifier/MaxPool2d[pool]/502(ImageClassifier::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #ImageClassifier::ImageClassifier/Conv2d[conv3]/ret.11(ImageClassifier::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #ImageClassifier::ImageClassifier/ReLU[relu]/ret.13(ImageClassifier::nndct_relu_8)
        self.module_9 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #ImageClassifier::ImageClassifier/MaxPool2d[pool]/540(ImageClassifier::nndct_maxpool_9)
        self.module_10 = py_nndct.nn.Module('nndct_reshape') #ImageClassifier::ImageClassifier/ret.15(ImageClassifier::nndct_reshape_10)
        self.module_11 = py_nndct.nn.Linear(in_features=2048, out_features=512, bias=True) #ImageClassifier::ImageClassifier/Linear[fc1]/ret.17(ImageClassifier::nndct_dense_11)
        self.module_12 = py_nndct.nn.ReLU(inplace=False) #ImageClassifier::ImageClassifier/ReLU[relu]/ret.19(ImageClassifier::nndct_relu_12)
        self.module_13 = py_nndct.nn.Linear(in_features=512, out_features=10, bias=True) #ImageClassifier::ImageClassifier/Linear[fc2]/ret(ImageClassifier::nndct_dense_13)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_0 = self.module_7(output_module_0)
        output_module_0 = self.module_8(output_module_0)
        output_module_0 = self.module_9(output_module_0)
        output_module_0 = self.module_10(input=output_module_0, shape=[-1,2048])
        output_module_0 = self.module_11(output_module_0)
        output_module_0 = self.module_12(output_module_0)
        output_module_0 = self.module_13(output_module_0)
        return output_module_0
