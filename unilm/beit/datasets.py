# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import os
import torch

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic
from timm.data import create_transform

from dall_e.utils import map_pixels
from masking_generator import MaskingGenerator
from dataset_folder import ImageFolder
from PIL import Image
from skimage.transform import resize

Image.LOAD_TRUNCATED_IMAGES = True

CIFAR10_DEFAULT_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_DEFAULT_STD = (0.24703223, 0.24348513, 0.26158784)

class DataAugmentationForBEiT(object):
    def __init__(self, args):
        
        if args.data_set == 'CIFAR10':
            mean = CIFAR10_DEFAULT_MEAN
            std = CIFAR10_DEFAULT_MEAN
        elif args.data_set == 'IMNET':
            imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
            mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
            std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        elif args.data_set == 'Retina':
            mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
            # mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
        
        if args.data_set == 'CIFAR10' or args.data_set == 'IMNET':
            self.common_transform = transforms.Compose([
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ])
        elif args.data_set == 'Retina' or args.data_set == 'COVIDx':
            '''https://github.com/xmengli/self_supervised/blob/master/main.py'''
            self.common_transform = transforms.Compose([
                # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=args.input_size, second_size=args.second_input_size,
                    scale=(0.2, 1.0),
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ])
        
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        if args.discrete_vae_type == "dall-e":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                map_pixels,
            ])
        elif args.discrete_vae_type == "customized":
            self.visual_token_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std),
                ),
            ])
        else:
            raise NotImplementedError()

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )
    
    def __call__(self, image):
        for_patches, for_visual_tokens = self.common_transform(image)
        return \
            self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
            self.masked_position_generator()
    
    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

        
# def build_beit_pretraining_dataset(args):
#     transform = DataAugmentationForBEiT(args)
#     print("Data Aug = %s" % str(transform))
    
#     data_all = np.load(os.path.join('./data/', args.dataset + '.npy'), allow_pickle = True)
#     data_all = data_all.item()
#     self.data_all = data_all[args.split_type]
    
#     if args.data_set == 'CIFAR10' or args.data_set == 'CIFAR100':
#         self.data = self.data_all['data'][args.single_client]
#         self.labels = self.data_all['target'][args.single_client]
            
    #     return datasets.CIFAR10(args.data_path, train=True, transform=transform)
    # else:
    #     return ImageFolder(args.data_path, transform=transform)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, 
                                   download=True, transform=transform)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, 
                                   download=True, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    
    if args.data_set == 'CIFAR10':
        mean = CIFAR10_DEFAULT_MEAN
        std = CIFAR10_DEFAULT_MEAN
    elif args.data_set == 'IMNET':
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    elif args.data_set == 'Retina':
            mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    
    if args.data_set == 'CIFAR10':
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

        else:
            t = []
            size = int((256 / 224) * 224)
            t.append(
                transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))
            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(mean, std))
            transform = transforms.Compose(t)
            
    else:
        if is_train:
            if args.data_set == 'Retina':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.4, 1.)),
                    transforms.RandomRotation(degrees=10),
                    # transforms.RandomGrayscale(p=0.2),
                    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                    ])
            else:
                transform = transforms.Compose([
                    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                    ])
                        
        else:
            transform = transforms.Compose([
                transforms.Resize([args.input_size, args.input_size]),
                transforms.ToTensor(), 
                # transforms.Normalize(
                #     mean=torch.tensor(mean),
                #     std=torch.tensor(std))
                ])
    return transform