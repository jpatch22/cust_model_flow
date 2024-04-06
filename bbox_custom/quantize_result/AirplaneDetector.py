# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class AirplaneDetector(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(AirplaneDetector, self).__init__()
        self.module_0 = py_nndct.nn.Input() #AirplaneDetector::input_0(AirplaneDetector::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #AirplaneDetector::AirplaneDetector/Conv2d[conv1]/ret.3(AirplaneDetector::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #AirplaneDetector::AirplaneDetector/ret.5(AirplaneDetector::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #AirplaneDetector::AirplaneDetector/430(AirplaneDetector::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #AirplaneDetector::AirplaneDetector/Conv2d[conv2]/ret.7(AirplaneDetector::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #AirplaneDetector::AirplaneDetector/ret.9(AirplaneDetector::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #AirplaneDetector::AirplaneDetector/468(AirplaneDetector::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #AirplaneDetector::AirplaneDetector/Conv2d[conv3]/ret.11(AirplaneDetector::nndct_conv2d_7)
        self.module_8 = py_nndct.nn.ReLU(inplace=False) #AirplaneDetector::AirplaneDetector/ret.13(AirplaneDetector::nndct_relu_8)
        self.module_9 = py_nndct.nn.AdaptiveAvgPool2d(output_size=[16, 16]) #AirplaneDetector::AirplaneDetector/AdaptiveAvgPool2d[adaptive_pool]/508(AirplaneDetector::nndct_adaptive_avg_pool2d_9)
        self.module_10 = py_nndct.nn.Module('nndct_shape') #AirplaneDetector::AirplaneDetector/511(AirplaneDetector::nndct_shape_10)
        self.module_11 = py_nndct.nn.Module('nndct_reshape') #AirplaneDetector::AirplaneDetector/ret.17(AirplaneDetector::nndct_reshape_11)
        self.module_12 = py_nndct.nn.Linear(in_features=16384, out_features=2, bias=True) #AirplaneDetector::AirplaneDetector/Linear[fc_cls]/ret.19(AirplaneDetector::nndct_dense_12)
        self.module_13 = py_nndct.nn.Linear(in_features=16384, out_features=4, bias=True) #AirplaneDetector::AirplaneDetector/Linear[fc_bb]/ret(AirplaneDetector::nndct_dense_13)

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
        output_module_10 = self.module_10(input=output_module_0, dim=0)
        output_module_11 = self.module_11(input=output_module_0, shape=[output_module_10,-1])
        output_module_12 = self.module_12(output_module_11)
        output_module_13 = self.module_13(output_module_11)
        return (output_module_12,output_module_13)
