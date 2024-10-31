import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter
from model import Model
import yaml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(opt):
    """Instantiate the model and count the total number of parameters."""
    # Create label converter
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    # Instantiate the model
    model = Model(opt)
    total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters in the model:', total_params)


def convert_types(opt):
    """Convert option types to the correct format."""
    opt.manualSeed = int(opt.manualSeed)
    opt.workers = int(opt.workers)
    opt.batch_size = int(opt.batch_size)
    opt.num_iter = int(opt.num_iter)
    opt.valInterval = int(opt.valInterval)
    opt.lr = float(opt.lr)
    opt.beta1 = float(opt.beta1)
    opt.rho = float(opt.rho)
    opt.eps = float(opt.eps)
    opt.grad_clip = float(opt.grad_clip)
    opt.batch_max_length = int(opt.batch_max_length)
    opt.imgH = int(opt.imgH)
    opt.imgW = int(opt.imgW)
    opt.num_fiducial = int(opt.num_fiducial)
    opt.input_channel = int(opt.input_channel)
    opt.output_channel = int(opt.output_channel)
    opt.hidden_size = int(opt.hidden_size)
    return opt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to the config file')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    # Set default save_path if not specified
    if not hasattr(opt, 'save_path'):
        opt.save_path = './saved_models'  # Default path

    opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
    os.makedirs(f'{opt.save_path}/{opt.exp_name}', exist_ok=True)

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    if opt.num_gpu > 1:
        opt.workers *= opt.num_gpu
        opt.batch_size *= opt.num_gpu

    opt = convert_types(opt)
    count_parameters(opt)
