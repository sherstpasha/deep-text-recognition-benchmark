import os
import sys
import re
import six
import math
import lmdb
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
import random
import pandas as pd
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform


class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        # print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        # log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        # assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset = OCRDataset(root=opt.train_data, opt=opt)
            total_number_dataset = len(_dataset)
            log.write('')

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = OCRDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log

# Функция для генерации случайного нечетного числа
def get_random_odd_int(min_value, max_value):
    value = random.randint(min_value, max_value)
    if value % 2 == 0:
        value += 1
    return value

# Функция для создания эффекта сепии
class Sepia(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Sepia, self).__init__(always_apply, p)

    def apply(self, img, **params):
        img = cv2.transform(img, np.matrix([[0.393, 0.769, 0.189],
                                            [0.349, 0.686, 0.168],
                                            [0.272, 0.534, 0.131]]))
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

# Функция для создания черно-белого изображения
class ToGrayScale(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(ToGrayScale, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Функция для добавления случайных кривых черточек
class RandomLines(ImageOnlyTransform):
    def __init__(self, num_lines=5, min_length=10, max_length=100, thickness=1, always_apply=False, p=0.5):
        super(RandomLines, self).__init__(always_apply, p)
        self.num_lines = num_lines
        self.min_length = min_length
        self.max_length = max_length
        self.thickness = thickness

    def apply(self, img, **params):
        img = img.copy()
        h, w, _ = img.shape
        for _ in range(self.num_lines):
            x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(self.min_length, self.max_length)
            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))
            color = (0, 0, 0)
            img = cv2.line(img, (x1, y1), (x2, y2), color, self.thickness)
        return img

# Функция для утолщения текста
class ThickenText(ImageOnlyTransform):
    def __init__(self, kernel_size=2, iterations=1, always_apply=False, p=0.5):
        super(ThickenText, self).__init__(always_apply, p)
        self.kernel_size = kernel_size
        self.iterations = iterations

    def apply(self, img, **params):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        img = cv2.dilate(img, kernel, iterations=self.iterations)
        return img

# Функция для утончения текста
class ThinText(ImageOnlyTransform):
    def __init__(self, kernel_size=2, iterations=1, always_apply=False, p=0.5):
        super(ThinText, self).__init__(always_apply, p)
        self.kernel_size = kernel_size
        self.iterations = iterations

    def apply(self, img, **params):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        img = cv2.erode(img, kernel, iterations=self.iterations)
        return img
    


# OCRDataset class continued
class OCRDataset(Dataset):
    def __init__(self, gt_file, opt):
        self.opt = opt
        self.data = pd.read_csv(gt_file, header=None, names=['path', 'label'])
        self.data.dropna(inplace=True)  # Удаляем строки с NaN значениями
        self.data.reset_index(drop=True, inplace=True)  # Обновляем индексы
        self.nSamples = len(self.data)
        print("nSamples1:", self.nSamples)
        if self.opt.data_filtering_off:
            self.filtered_index_list = [index for index in range(self.nSamples)]
        else:
            self.filtered_index_list = []
            for index in range(self.nSamples):
                label = self.data.at[index, 'label']
                if len(label) > self.opt.batch_max_length:
                    continue
                out_of_char = f'[^{self.opt.character}]'
                if re.search(out_of_char, label.lower()):
                    continue
                self.filtered_index_list.append(index)
            self.nSamples = len(self.filtered_index_list)

        print("nSamples:", self.nSamples)
        self.current_index = 0

        # Define augmentations
        self.augmentations = A.Compose([
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.RandomRain(drop_length=5, drop_width=1, drop_color=(255, 255, 255), blur_value=2, p=0.3),
            A.RandomRain(drop_length=10, drop_width=2, drop_color=(0, 0, 0), blur_value=2, p=0.3),
            A.RandomShadow(p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
            A.InvertImg(p=0.3),
            A.Solarize(threshold=128, p=0.3),
            Sepia(p=0.5),
            ToGrayScale(p=0.5),
            RandomLines(num_lines=5, min_length=20, max_length=100, thickness=2, p=0.5),
            ThickenText(kernel_size=2, iterations=2, p=0.5),
            ThinText(kernel_size=2, iterations=2, p=0.5)
        ])

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        index = self.filtered_index_list[index]
        img_path = self.data.at[index, 'path']
        label = self.data.at[index, 'label']

        img = self.load_image(img_path)

        if not self.opt.sensitive:
            label = label.lower()

        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)

        img = np.array(img)
        augmented = self.augmentations(image=img)
        img = augmented['image']

        return (img, label)

    def load_image(self, img_path):
        try:
            if self.opt.rgb:
                img = Image.open(img_path).convert('RGB')
            else:
                img = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            random_img_path = random.choice(self.data['path'])
            img = self.load_image(random_img_path)
        return img

    def get_batch(self, batch_size):
        batch_images = []
        batch_texts = []

        end_index = min(self.current_index + batch_size, self.nSamples)
        indices = range(self.current_index, end_index)

        for index in indices:
            img, label = self.__getitem__(index)
            batch_images.append(img)
            batch_texts.append(label)

        self.current_index = end_index % self.nSamples  # Loop around if we reach the end

        transform = ResizeNormalize((self.opt.imgW, self.opt.imgH))
        batch_images = [transform(img) for img in batch_images]
        batch_images = torch.stack(batch_images, 0)

        return batch_images, batch_texts

class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
