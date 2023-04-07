# --------------------------------------------------------
# Data Augmentation techniques for Fed-BEiT and Fed-MAE during 
# pre-training and fine-tuning.
# --------------------------------------------------------'
import torch

from torchvision import transforms
from .transforms import RandomResizedCropAndInterpolationWithTwoPic
from .dall_e.utils import map_pixels
from .masking_generator import MaskingGenerator

from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True


RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)

class DataAugmentationForPretrain(object):
    """ data transformations for pre-training"""
    def __init__(self, args):

        if args.data_set == 'Retina':
            mean, std = RETINA_MEAN, RETINA_STD
        else:   
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
        if args.model_name == 'beit':
            if args.data_set == 'Retina':
                self.common_transform = transforms.Compose([
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=args.input_size, second_size=args.second_input_size,
                        scale=(0.2, 1.0),
                        interpolation=args.train_interpolation,
                        second_interpolation=args.second_interpolation,
                    ),
                ])
            elif args.data_set == 'COVID-FL':
                self.common_transform = transforms.Compose([
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=args.input_size, second_size=args.second_input_size,
                        scale=(0.4, 1.0),
                        interpolation=args.train_interpolation,
                        second_interpolation=args.second_interpolation,
                    ),
                ])
            else:
                self.common_transform = transforms.Compose([
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5),
                    RandomResizedCropAndInterpolationWithTwoPic(
                        size=args.input_size, second_size=args.second_input_size,
                        scale=(0.2, 1.0),
                        interpolation=args.train_interpolation,
                        second_interpolation=args.second_interpolation,
                    ),
                ])
            
            # visual_token_transform
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
            
            args.num_mask_patches = int(round((args.mask_ratio * 196.0)/5.0)*5.0) 

            self.masked_position_generator = MaskingGenerator(
                args.window_size, num_masking_patches=args.num_mask_patches,
                max_num_patches=args.max_mask_patches_per_block,
                min_num_patches=args.min_mask_patches_per_block,
            )
        
        elif args.model_name == 'mae':
            if args.data_set == 'Retina':
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic   
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5)])

            elif args.data_set == 'COVID-FL':
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.4, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.ColorJitter(hue=.05, saturation=.05),
                    transforms.RandomHorizontalFlip(p=0.5)])
            else:
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomHorizontalFlip(p=0.5)])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        
        self.args = args
    
    def __call__(self, image):
        if self.args.model_name == 'beit':
            for_patches, for_visual_tokens = self.common_transform(image)
            return \
                self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
                self.masked_position_generator()
        elif self.args.model_name == 'mae':
            for_patches = self.common_transform(image)
            return self.patch_transform(for_patches)

    def __repr__(self):
        if self.args.model_name == 'beit':
            repr = "(DataAugmentationForBEiT,\n"
            repr += "  common_transform = %s,\n" % str(self.common_transform)
            repr += "  patch_transform = %s,\n" % str(self.patch_transform)
            repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
            repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
            repr += ")"
            
        elif self.args.model_name == 'mae':
            repr = "(DataAugmentationFoMAE,\n"
            repr += "  common_transform = %s,\n" % str(self.common_transform)
            repr += "  patch_transform = %s,\n" % str(self.patch_transform)

        return repr



def build_transform(is_train, mode, args):
    """ data transformations for fine-tuning"""

    if args.data_set == 'Retina':
        mean, std = RETINA_MEAN, RETINA_STD
    else:   
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    
    if mode == 'finetune':
        if is_train:
            if args.data_set == 'COVID-FL':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.2)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                    ])
            else:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size, scale=(0.6, 1.)),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std))
                    ])

        else:
            transform = transforms.Compose([
                transforms.Resize(size=args.input_size),
                transforms.CenterCrop(size=(args.input_size, args.input_size)), 
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
                ])

    return transform
