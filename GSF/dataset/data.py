import logging
import math
import random

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

stl10_mean = (0.4408, 0.4279, 0.3867)
stl10_std = (0.2683, 0.2610, 0.2687)
svhn_mean = (0.4377, 0.4438, 0.4728)
svhn_std = (0.1980, 0.2010, 0.1970)
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
imagenet_mean=(0.485, 0.456, 0.406)
imagenet_std=(0.229, 0.224, 0.225)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_stl10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=96,
                              padding=int(96*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stl10_mean, std=stl10_std)
    ])
    
    train_labeled_dataset = datasets.STL10(
        args.root, split='train', folds=args.seed-1, 
        transform=transform_labeled, download=True)

    train_unlabeled_dataset = datasets.STL10(
        args.root, split='train+unlabeled', 
        transform=TransformFixMatch(mean=stl10_mean, std=stl10_std, data='stl10'))

    test_dataset = datasets.STL10(
        args.root, split='test', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_svhn(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=svhn_mean, std=svhn_std)
    ])
    base_dataset = datasets.SVHN(args.root, split='train', download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.labels)

    train_labeled_dataset = SVHNSSL(
        args.root, train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset = SVHNSSL(
        args.root, train_unlabeled_idxs, split='train',
        transform=TransformFixMatch(mean=svhn_mean, std=svhn_std, data='svhn'))

    test_dataset = datasets.SVHN(
        args.root, split='test', transform=transform_val, download=True)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar10(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(args.root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        args.root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        args.root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        args.root, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        args.root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        args.root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        args.root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        args.root, train=False, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_imagenet(args):
    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    base_dataset = datasets.ImageNet(args.root, split='train') # pre-requisite: 'wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz'

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = ImageNetSSL(
        args.root, train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset = ImageNetSSL(
        args.root, train_unlabeled_idxs, split='train',
        transform=TransformFixMatch(mean=imagenet_mean, std=imagenet_std, data='imagenet'))

    test_dataset = datasets.ImageNet(args.root, split='val', transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset
    

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]
        idx = np.random.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std, data='cifar'):
        if data == 'imagenet':
            self.weak = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip()])
            self.strong = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                RandAugmentMC(n=2, m=10)])
        elif data == 'cifar' or data == 'svhn':
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32*0.125),
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32,
                                      padding=int(32*0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
        elif data == 'stl10':
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=96,
                                      padding=int(96*0.125),
                                      padding_mode='reflect')])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=96,
                                      padding=int(96*0.125),
                                      padding_mode='reflect'),
                RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs].transpose(0, 2, 3, 1)
            self.labels = self.labels[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class ImageNetSSL(datasets.ImageNet):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=None):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.samples = [self.samples[i] for i in indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
        

DATASET_GETTERS = {'stl10': get_stl10, 
                   'svhn': get_svhn,
                   'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'imagenet': get_imagenet,}
