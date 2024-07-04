import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import yaml
import argparse
import string

from utils import CTCLabelConverter, AttnLabelConverter
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(config_path, model_path):
    """ Load the trained model """
    with open(config_path, 'r', encoding='utf-8') as f:
        opt = argparse.Namespace(**yaml.safe_load(f))

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(device)

    print(f'Loading pretrained model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, converter, opt

def preprocess_image(image_path, opt):
    """ Preprocess the input image """
    img = Image.open(image_path)
    if not opt.rgb:
        img = img.convert('L')

    transform = transforms.Compose([
        transforms.Resize((opt.imgH, opt.imgW), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(img).unsqueeze(0)  # Add batch dimension
    return img.to(device)

def recognize_text(model, converter, opt, image_path):
    """ Recognize text from the preprocessed image """
    image = preprocess_image(image_path, opt)
    length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
    text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)

    if 'CTC' in opt.Prediction:
        preds = model(image, text_for_pred)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, torch.IntTensor([preds.size(1)]))
    else:
        preds = model(image, text_for_pred, is_train=False)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, length_for_pred)

    return preds_str

