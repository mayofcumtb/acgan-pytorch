# -*- coding: utf-8 -*-
import torch.utils.data as data
from PIL import Image
import os
import os.path

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
def read_annotation_from_filename(fname):
    # e.g. A sample of annotation filname path = '02691156_10155655850468db78d106ce0a280f87_a008_e-29_t012_d001.png
    labels = fname[:-4].split('_')
    azimuth = int(labels[2][1:])
    elevation = int(labels[3][1:])
    tilt = int(labels[4][1:])
    distance = int(labels[5][1:])
    return azimuth, elevation, tilt, distance

def make_syn_datasets(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir,target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    azimuth, elevation, tilt, distance = read_annotation_from_filename(fname)
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target], azimuth, elevation, tilt, distance)
                    images.append(item)
    return images
def make_real_datasets(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def check_files_extensions(samples, path, extensions):
    if len(samples) == 0:
        raise (RuntimeError("Found 0 files in subfolders of" + path + "\n"
                                                                      "supported extensions are:" + ",".join(
            extensions)))

class SynDatasetFolder(data.Dataset):
    """
    modify class code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py,
    Because we need more annotations from the file
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):



        self.root = root
        self.dir_syn = os.path.join(self.root, 'syn')
        self.dir_real = os.path.join(self.root, 'real')


        classes, class_to_idx = find_classes(self.dir_real)

        syn_sample = make_syn_datasets(self.dir_syn)
        real_sample = make_real_datasets(self.dir_real)

        check_files_extensions(syn_sample)
        check_files_extensions(real_sample)

        self.syn_size = len(self.dir_syn)
        self.real_size = len(self.dir_real)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.syn_sample = syn_sample
        self.real_sample = real_sample

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        syn_path, syn_class_index, azimuth, elevation, tilt, distance = self.syn_sample[index]
        real_path, real_class_index = self.real_sample[index]

        syn_image, syn_alpha = self.loader(syn_path)
        real_image = Image.open(real_path).convert('RGB')
        if self.transform is not None:
            syn_image = self.transform(syn_image)
            syn_alpha = self.transform(syn_alpha)
            real_image = self.transform(real_image)
        if self.target_transform is not None:
            syn_image = self.target_transform(syn_image)
            syn_alpha = self.target_transform(syn_alpha)
            real_image = self.target_transform(real_image)
        return syn_image, real_image, syn_alpha, syn_class_index, real_class_index, azimuth, elevation, tilt, distance


    def __len__(self):
        return max(self.syn_size, self.real_size)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    def name(self):
        return 'SynDatasetFolder'

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def color_gray_loader(path):
    # open path as file to avoid ResourceWarning( https://github.com/python-pillow/issue/835
    with open(path, 'rb') as f:
        img = Image.open(f)
        r, g, b, a = img.split()
        color_img = Image.merge("RGB", (r, g, b))

        return color_img, a.convert('L')

class SynImageDatasets(SynDatasetFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=color_gray_loader):
        super(SynImageDatasets, self).__init__(root, loader, IMG_EXTENSIONS, transform=transform,
                                               target_transform=target_transform)
    def name(self):
        return "SynImageDatasets"