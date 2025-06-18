
import os
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np



class ImageNetFFCV:
    """
    Relevant DataLoaders and other functions for training on ImageNet.
    Uses FFCV. Please ensure that it is installed!

    Args:
        data_root (str): path to root folder which contains the train and validation betons.
        Ensure that the file name is the same as in the class.
        batch_size (int): batch size you'd like to train with
        distributed (bool, optional): whether you'd like to train on multiple GPUs or not, by default: False
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256

    def __init__(self, args):
        super(ImageNetFFCV, self).__init__()

        data_root = os.path.join(args.data, "imagenet")  # Match CIFAR path structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = args.batch_size
        num_workers = args.workers
        distributed = getattr(args, 'distributed', False)

        

        try:
            from ffcv.loader import Loader, OrderOption
            from ffcv.transforms import (
                ToTensor,
                ToDevice,
                Squeeze,
                NormalizeImage,
                RandomHorizontalFlip,
                ToTorchImage,
            )
            from ffcv.fields.decoders import (
                RandomResizedCropRGBImageDecoder,
                CenterCropRGBImageDecoder,
            )
            from ffcv.fields.basics import IntDecoder
            print('FFCV loaded, all good -- you can work on ImageNet!')
        except ImportError:
            raise ImportError("FFCV not installed! Follow installation instructions")

        train_pipeline = [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(self.device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(self.IMAGENET_MEAN, 
                          self.IMAGENET_STD, 
                          np.float32),
            ]

        val_pipeline = [
            CenterCropRGBImageDecoder((224, 224), ratio=self.DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice(torch.device(self.device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.IMAGENET_MEAN, 
                          self.IMAGENET_STD, 
                          np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        self.train_loader = Loader(
            os.path.join(data_root, "train_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=order,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

        self.val_loader = Loader(
            os.path.join(data_root, "val_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": val_pipeline, "label": label_pipeline},
            distributed=distributed,
        )

class imagenet_subsampled:
    """
    Relevant DataLoaders and other functions for training on ImageNet.
    Uses FFCV. Please ensure that it is installed!
    Subsamples and uses a random subset of the dataset.
    Args:
        data_root (str): path to root folder which contains the train and validation betons.
        Ensure that the file name is the same as in the class.
        batch_size (int): batch size you'd like to train with
        distributed (bool, optional): whether you'd like to train on multiple GPUs or not, by default: False
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224 / 256


    def __init__(self, args):

        super(ImageNetFFCV, self).__init__()

        data_root = os.path.join(args.data, "imagenet")  # Match CIFAR path structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        batch_size = args.batch_size
        num_workers = args.workers
        subsample_frac = getattr(args, 'subsample_frac', 0.1)  # Default to 10% if not specified
        
        distributed = getattr(args, 'distributed', False)

        # imagenet has these many train images
        
        try:
            from ffcv.loader import Loader, OrderOption
            from ffcv.transforms import (
                ToTensor,
                ToDevice,
                Squeeze,
                NormalizeImage,
                RandomHorizontalFlip,
                ToTorchImage,
            )
            from ffcv.fields.decoders import (
                RandomResizedCropRGBImageDecoder,
                CenterCropRGBImageDecoder,
            )
            from ffcv.fields.basics import IntDecoder
            print('FFCV loaded, all good -- you can work on ImageNet!')
        except ImportError:
            raise ImportError(
                'FFCV is not installed. Please install FFCV to train on ImageNet. '
                'You can install it by following the instructions provided in the repository'
            )
        
        train_pipeline = [
                RandomResizedCropRGBImageDecoder((224, 224)),
                RandomHorizontalFlip(),
                ToTensor(),
                ToDevice(torch.device(self.device), non_blocking=True),
                ToTorchImage(),
                NormalizeImage(self.IMAGENET_MEAN, 
                          self.IMAGENET_STD, 
                          np.float32),
            ]

        val_pipeline = [
            CenterCropRGBImageDecoder((224, 224), ratio=self.DEFAULT_CROP_RATIO),
            ToTensor(),
            ToDevice(torch.device(self.device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(self.IMAGENET_MEAN, 
                          self.IMAGENET_STD, 
                          np.float32),
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.device), non_blocking=True)
        ]

        num_images = 1281167
        indices = list(range(num_images))
        # # Shuffle indices and select a subset
        np.random.shuffle(indices)
        indices = indices[:int(subsample_frac * num_images)]

        print('Subsampled indiced are being chosen')
        self.train_loader = Loader(
            os.path.join(data_root, "train_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM,
            os_cache=True,
            drop_last=True,
            pipelines={"image": train_pipeline, "label": label_pipeline},
            distributed=distributed,
            indices=indices
        )
        

        self.val_loader = Loader(
            os.path.join(data_root, "val_500_0.50_90.beton"),
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.SEQUENTIAL,
            drop_last=False,
            pipelines={"image": val_pipeline, "label": label_pipeline},
            distributed=distributed,
        )