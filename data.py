import os
import torch
from torchvision.transforms import transforms
from torchvision import datasets

from data_augmentation import TwoCropsTransform


class Data:
    def __init__(self, args):
        self.args = args
        self.train_dir = os.path.join(args.data_dir_path, 'train')
        self.val_dir = os.path.join(args.data_dir_path, 'val')
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_img_transforms = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)]),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        self.normalize,
                    ])

        self.val_img_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        self.normalize])

    def get_data_loaders(self):
        train_two_crops_transform = TwoCropsTransform(self.train_img_transforms)
        val_two_crops_transform = TwoCropsTransform(self.val_img_transforms)
        train_dataset = datasets.ImageFolder(
            self.train_dir, train_two_crops_transform)
        val_dataset = datasets.ImageFolder(self.val_dir, val_two_crops_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.args.batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=1)
        return train_loader, val_loader
