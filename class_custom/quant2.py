import os
import re
import sys
import argparse
import time
import pdb
import random
from pytorch_nndct.apis import torch_quantizer
import torch
import numpy as np
from imageClassifier import ImageClassifier


num_samples = 1
num_channels = 3
image_height = 32
image_width = 32
num_classes = 10
input_shape = (num_channels, image_height, image_width)
def evaluate(model, device, dataset):
    model.eval()
    for inp in dataset:
        out = model(dataset)


def main():
    inputs = torch.randn([num_samples, *input_shape])
    quant_mode = "test"
    device = torch.device("cpu")
    config_file=None
    target=None
    model = ImageClassifier(image_height, image_width, num_classes)
    model.load_state_dict(torch.load("image_classifier.pth"))
    quantizer = torch_quantizer(
                    quant_mode, model, (inputs), device=device, quant_config_file=config_file, target=target)
    quant_model = quantizer.quant_model
    evaluate(quant_model, device, torch.randn([num_samples, *(input_shape)]))

    if quant_mode == "calib":
        quantizer.export_quant_config()
    else:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel(deploy_check=False)

if __name__ == "__main__":
    main()
