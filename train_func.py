import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import (
    CTCLabelConverter,
    CTCLabelConverterForBaiduWarpctc,
    AttnLabelConverter,
    Averager,
)
from dataset import OCRDataset2, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
import yaml
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    train_csv,
    train_root,
    valid_csv,
    valid_root,
    batch_size,
    num_iter,
    valInterval,
    saved_model,
    output_dir,
):
    class Config:
        pass

    opt = Config()
    opt.exp_name = output_dir
    opt.manualSeed = 1111
    opt.workers = 0
    opt.batch_size = batch_size
    opt.num_iter = num_iter
    opt.valInterval = valInterval
    opt.saved_model = saved_model
    opt.FT = True
    opt.adam = False
    opt.lr = 1.0
    opt.beta1 = 0.9
    opt.rho = 0.95
    opt.eps = 1e-8
    opt.grad_clip = 5
    opt.baiduCTC = False
    opt.select_data = "MJ-ST"
    opt.batch_ratio = "1"
    opt.total_data_usage_ratio = 1.0
    opt.batch_max_length = 24
    opt.imgH = 32
    opt.imgW = 128
    opt.rgb = False
    opt.character = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя!\"#$%&'()*+,-./:;<=>?@[\\]№_`{|}~ €₽"
    opt.sensitive = True
    opt.PAD = False
    opt.data_filtering_off = False
    opt.Transformation = "TPS"
    opt.FeatureExtraction = "ResNet"
    opt.SequenceModeling = "BiLSTM"
    opt.Prediction = "Attn"
    opt.num_fiducial = 20
    opt.input_channel = 1
    opt.output_channel = 512
    opt.hidden_size = 256
    opt.train_root = train_root
    opt.train_csv = train_csv
    opt.valid_root = valid_root
    opt.valid_csv = valid_csv

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

    print("Filtering the images containing characters which are not in opt.character")
    print("Filtering the images whose label is longer than opt.batch_max_length")

    opt.batch_ratio = opt.batch_ratio.split("-")
    train_dataset = OCRDataset2(opt.train_csv, opt.train_root, opt)

    # Ensure the directory exists
    os.makedirs(f"./saved_models/{opt.exp_name}", exist_ok=True)

    # Open the log file for appending
    log = open(f"./saved_models/{opt.exp_name}/log_dataset.txt", "a")
    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    valid_dataset = OCRDataset2(opt.valid_csv, opt.valid_root, opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )
    log.write("-")
    print("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()

    AlignCollate_train = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_train,
        pin_memory=True,
    )

    """ model configuration """
    if "CTC" in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print(
        "model input parameters",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )

    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception as e:
            if "weight" in name:
                param.data.fill_(1)
            continue

    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != "":
        print(f"loading pretrained model from {opt.saved_model}")
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    loss_avg = Averager()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print("Trainable params num : ", sum(params_num))

    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps
        )
    print("Optimizer:")
    print(optimizer)

    with open(
        f"./saved_models/{opt.exp_name}/opt.txt", "a", encoding="utf8"
    ) as opt_file:
        opt_log = "------------ Options -------------\n"
        args = vars(opt)
        for k, v in args.items():
            opt_log += f"{str(k)}: {str(v)}\n"
        opt_log += "---------------------------------------\n"
        print(opt_log)
        opt_file.write(opt_log)

    start_iter = 0
    if opt.saved_model != "":
        try:
            start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
            print(f"continue to train, start_iter: {start_iter}")
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    pbar = tqdm(total=opt.valInterval, desc="Training Progress", leave=False)

    for epoch in range(opt.num_iter):
        for image_tensors, labels in train_loader:
            image = image_tensors.to(device)
            text, length = converter.encode(
                labels, batch_max_length=opt.batch_max_length
            )
            batch_size = image.size(0)

            if "CTC" in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                if opt.baiduCTC:
                    preds = preds.permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length) / batch_size
                else:
                    preds = preds.log_softmax(2).permute(1, 0, 2)
                    cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text[:, :-1])
                target = text[:, 1:]
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            loss_avg.add(cost)

            pbar.update(1)

            if (iteration + 1) % opt.valInterval == 0 or iteration == 0:
                pbar.close()
                elapsed_time = time.time() - start_time
                with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log:
                    model.eval()
                    with torch.no_grad():
                        (
                            valid_loss,
                            current_accuracy,
                            current_norm_ED,
                            preds,
                            confidence_score,
                            labels,
                            infer_time,
                            length_of_data,
                        ) = validation(model, criterion, valid_loader, converter, opt)
                    model.train()

                    loss_log = f"[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}"
                    loss_avg.reset()

                    current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                    if current_accuracy > best_accuracy:
                        best_accuracy = current_accuracy
                        torch.save(
                            model.state_dict(),
                            f"./saved_models/{opt.exp_name}/best_accuracy.pth",
                        )
                    if current_norm_ED > best_norm_ED:
                        best_norm_ED = current_norm_ED
                        torch.save(
                            model.state_dict(),
                            f"./saved_models/{opt.exp_name}/best_norm_ED.pth",
                        )
                    best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                    loss_model_log = (
                        f"{loss_log}\n{current_model_log}\n{best_model_log}"
                    )
                    print(loss_model_log)
                    log.write(loss_model_log + "\n")

                    dashed_line = "-" * 80
                    head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                    predicted_result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
                    for gt, pred, confidence in zip(
                        labels[:10], preds[:10], confidence_score[:10]
                    ):
                        if "Attn" in opt.Prediction:
                            gt = gt[: gt.find("[s]")]
                            pred = pred[: pred.find("[s]")]

                        predicted_result_log += f"{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n"
                    predicted_result_log += f"{dashed_line}"
                    print(predicted_result_log)
                    log.write(predicted_result_log + "\n")

                pbar = tqdm(
                    total=opt.valInterval, desc="Training Progress", leave=False
                )

            if (iteration + 1) % 1e5 == 0:
                torch.save(
                    model.state_dict(),
                    f"./saved_models/{opt.exp_name}/iter_{iteration+1}.pth",
                )

            if (iteration + 1) == opt.num_iter:
                print("end the training")
                sys.exit()
            iteration += 1
