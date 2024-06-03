from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.datasets.utils import check_integrity
from typing import *
from zipdata import ZipData

import bisect 
import numpy as np
import os, glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import torch

# set this environment variable to the location of your imagenet directory if you want to read ImageNet data.
# make sure your val directory is preprocessed to look like the train directory, e.g. by running this script
# https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
IMAGENET_LOC_ENV = "IMAGENET_DIR"

# list of all datasets
DATASETS = ["imagenet", "imagenet32", "cifar10", "pneumonia", "breakhis", "isic", "hyper"]

def get_dataset(dataset: str, split: str, data_dir: str, noise_sd=0) -> Dataset:
    """Return the dataset as a PyTorch Dataset object"""
    if dataset == "imagenet":
        return _imagenet(split)
    
    elif dataset == "imagenet32":
        return _imagenet32(split)

    elif dataset == "cifar10":
        return _cifar10(split)

    elif dataset == "pneumonia":
        return _pneumonia(split)

    elif dataset == "breakhis":
        return _breakhis(split)

    elif dataset == "isic":
        return _isic(split)

    elif dataset == "hyper":
        return _hyper(split, data_dir, noise_sd)


def get_num_classes(dataset: str):
    """Return the number of classes in the dataset. """
    if dataset == "imagenet":
        return 1000
    elif dataset == "cifar10":
        return 10
    elif dataset == "pneumonia":
        return 2
    elif dataset == "breakhis":
        return 2
    elif dataset == "isic":
        return 7
    elif dataset == "hyper":
        return 23

def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == "imagenet":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    elif dataset == "cifar10":
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)
    elif dataset == "imagenet32":
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)
    else:
        return NormalizeLayer(_IMAGENET_MEAN, _IMAGENET_STDDEV)

def get_input_center_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's Input Centering layer"""
    if dataset == "imagenet":
        return InputCenterLayer(_IMAGENET_MEAN)
    elif dataset == "cifar10":
        return InputCenterLayer(_CIFAR10_MEAN)
    else:
        return InputCenterLayer(_IMAGENET_MEAN)


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STDDEV = [0.229, 0.224, 0.225]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


def _cifar10(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'dataset_cache')
    if split == "train":
        return datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]))
    elif split == "test":
        return datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.ToTensor())

    else:
        raise Exception("Unknown split name.")


def _imagenet(split: str) -> Dataset:
    if not IMAGENET_LOC_ENV in os.environ:
        raise RuntimeError("environment variable for ImageNet directory not set")

    dir = os.environ[IMAGENET_LOC_ENV]
    if split == "train":
        subdir = os.path.join(dir, "train")
        transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    elif split == "test":
        subdir = os.path.join(dir, "val")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    return datasets.ImageFolder(subdir, transform)


def _imagenet32(split: str) -> Dataset:
    dataset_path = os.path.join(os.getenv('PT_DATA_DIR', 'datasets'), 'Imagenet32')
   
    if split == "train":
        return ImageNetDS(dataset_path, 32, train=True, transform=transforms.Compose([
                        transforms.RandomCrop(32, padding=4), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]))
    
    elif split == "test":
        return ImageNetDS(dataset_path, 32, train=False, transform=transforms.ToTensor())

def _breakhis(split: str) -> Dataset:
    data_dir = '../../X-MONAI/data/BreaKHis_v1/'
    if split == "train":
        return Breakhis(root=data_dir, transform=transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.RandomCrop(224), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]), split='train')
    elif split == "test":
        return Breakhis(root=data_dir, transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()]), split='test')

def _pneumonia(split: str) -> Dataset:
    data_dir = "../../X-MONAI/data/chest_xray/chest_xray"
    if split == "train":
        return datasets.ImageFolder(os.path.join(data_dir, 'train'),transform = transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.RandomCrop(224), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]))
    elif split == "test":
        return datasets.ImageFolder(os.path.join(data_dir, 'test'),transform = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()]))


def _isic(split: str) -> Dataset:
    data_dir = '../../X-MONAI/data'
    if split == "train":
        return ISIC(root=data_dir, transform=transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.RandomCrop(224), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor()]), split='train')
    
    elif split == "test":
        return ISIC(root=data_dir ,transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor()]), split='test')

def _hyper(split: str, data_dir, noise_sd) -> Dataset:
    # data_dir = '../../X-MONAI/data'
   
    if split == "train":
        return Hyper(root=data_dir, train=True, transform=transforms.Compose([
                        transforms.Resize((256,256)),
                        transforms.RandomCrop(224), 
                        transforms.RandomHorizontalFlip(), 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV)]), noise_sd=noise_sd)
    
    elif split == "test":
        return Hyper(root=data_dir, train=False, transform=transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV),]), noise_sd=noise_sd)

class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means)/sds


# from https://github.com/hendrycks/pre-training
class ImageNetDS(Dataset):
    """`Downsampled ImageNet <https://patrykchrabaszcz.github.io/Imagenet32/>`_ Datasets.

    Args:
        root (string): Root directory of dataset where directory
            ``ImagenetXX_train`` exists.
        img_size (int): Dimensions of the images: 64,32,16,8
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'Imagenet{}_train'
    train_list = [
        ['train_data_batch_1', ''],
        ['train_data_batch_2', ''],
        ['train_data_batch_3', ''],
        ['train_data_batch_4', ''],
        ['train_data_batch_5', ''],
        ['train_data_batch_6', ''],
        ['train_data_batch_7', ''],
        ['train_data_batch_8', ''],
        ['train_data_batch_9', ''],
        ['train_data_batch_10', '']
    ]

    test_list = [
        ['val_data', ''],
    ]

    def __init__(self, root, img_size, train=True, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.img_size = img_size

        self.base_folder = self.base_folder.format(img_size)

        # if not self._check_integrity():
        #    raise RuntimeError('Dataset not found or corrupted.') # TODO

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                with open(file, 'rb') as fo:
                    entry = pickle.load(fo)
                    self.train_data.append(entry['data'])
                    self.train_labels += [label - 1 for label in entry['labels']]
                    self.mean = entry['mean']

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((self.train_data.shape[0], 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            self.test_data = entry['data']
            self.test_labels = [label - 1 for label in entry['labels']]
            fo.close()
            self.test_data = self.test_data.reshape((self.test_data.shape[0], 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


HYPER_CLASSES = {
    'cecum':0, 'ileum':1, 'retroflex-rectum':2, 'hemorrhoids':3, 'polyps':4,
       'ulcerative-colitis-grade-0-1':5, 'ulcerative-colitis-grade-1':6,
       'ulcerative-colitis-grade-1-2':7, 'ulcerative-colitis-grade-2':8,
       'ulcerative-colitis-grade-2-3':9, 'ulcerative-colitis-grade-3':10,
       'bbps-0-1':11, 'bbps-2-3':12, 'impacted-stool':13, 'dyed-lifted-polyps':14,
       'dyed-resection-margins':15, 'pylorus':16, 'retroflex-stomach':17, 'z-line':18,
       'barretts':19, 'barretts-short-segment':20, 'esophagitis-a':21,
       'esophagitis-b-d':22
    }

class Hyper(Dataset):
    def __init__(self, root='../../medical_image_experiments', train=True, transform=None, noise_sd=0):
        self.train = train
        self.image_paths = glob.glob(f'{root}/labeled-images/*/*/*/*.jpg')
        self.labels = pd.read_csv(f'{root}/labeled-images/image-labels.csv')
        self.split = pd.read_csv('https://raw.githubusercontent.com/simula/hyper-kvasir/master/official_splits/2_fold_split.csv',sep=';')
        self.get_split()
        self.noise_sd = noise_sd
        self.targets = []
        for image_path in self.image_paths:
            self.targets.append(HYPER_CLASSES[self.labels[self.labels['Video file'] == image_path.split('/')[-1].split('.')[0]]['Finding'].reset_index(drop=True)[0]])
        self.transform = transform

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)

    def __getitem__(self, i):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[i]
        image = Image.open(image_path).convert('RGB')

        target = self.targets[i] #HYPER_CLASSES[self.labels[self.labels['Video file'] == image_path.split('/')[-1].split('.')[0]]['Finding'].reset_index(drop=True)[0]]

        if self.transform is not None:
            image = self.transform(image)

        #add gaussian noise to the image
        noise = torch.randn_like(image) * self.noise_sd
        image = image + noise

        return image, target

    def get_split(self):
        idx = 0 if self.train else 1
        paths = list(self.split[self.split['split-index'] == idx]['file-name'])
        self.image_paths = [path for path in self.image_paths if path.split('/')[-1] in paths]

    def get_frequency(self):
        frequency = [0] * 23
        for image_path in self.image_paths:
            idx = HYPER_CLASSES[self.labels[self.labels['Video file'] == image_path.split('/')[-1].split('.')[0]]['Finding'].reset_index(drop=True)[0]]
            frequency[idx] += 1
        return frequency


class Breakhis(Dataset):
    def __init__(self, root='.', transform=None, split='train'):
        self.paths = glob.glob(f"{root}/BreaKHis_v1/histology_slides/breast/*/SOB/*/*/*/*.png")
        if split == 'train':
            self.paths, _ = train_test_split(self.paths, random_state= 1024, test_size=0.25)
        elif split == 'test':
            _, self.paths = train_test_split(self.paths, random_state= 1024, test_size=0.25)

        self.transform = transform
        self.label_dict = {'M' : 0, 'B' : 1}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image_path = self.paths[i]
        image = Image.open(image_path)

        target = self.label_dict[image_path.split('/')[-1].split('_')[1]]

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def get_frequency(self):
        frequency = [0, 0]
        for image_path in self.paths:
            idx = self.label_dict[image_path.split('/')[-1].split('_')[1]]
            frequency[idx] += 1
        return frequency
            

class ISIC(Dataset):
    def __init__(self, root='', transform=None, split='train'):
        if split == 'train':
            self.paths = glob.glob(f'{root}/ISIC2018_Task3_Training_Input/*jpg')
            self.label = pd.read_csv(f'{root}/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
            self.paths, _ = train_test_split(self.paths, random_state= 1024, test_size=0.25)
        elif split == 'test':
            self.paths = glob.glob(f'{root}/ISIC2018_Task3_Training_Input/*jpg')
            self.label = pd.read_csv(f'{root}/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
            _, self.paths = train_test_split(self.paths, random_state= 1024, test_size=0.25)
        # else:
        #     self.paths = glob.glob(f'{root}/ISIC2018_Task3_Validation_Input/*jpg')
        #     self.label = pd.read_csv(f'{root}/ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image_path = self.paths[i]
        image = Image.open(image_path)

        target = np.argmax(np.array(self.label[self.label['image'] == image_path.split('/')[-1].split('.')[0]].iloc[:,1:])).item()

        if self.transform is not None:
            image = self.transform(image)

        return image, target
    
    def get_frequency(self):
        frequency = [0] * 7
        for image_path in self.paths:
            idx = np.argmax(np.array(self.label[self.label['image'] == image_path.split('/')[-1].split('.')[0]].iloc[:,1:])).item()
            frequency[idx] += 1
        return frequency

if __name__ == "__main__":
    dataset = get_dataset('imagenet32', 'train')
    embed()

