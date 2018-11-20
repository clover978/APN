import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys

def make_dataset(listfile, class_to_idx, extensions):
    images = []
    with open(listfile) as f:
        for line in f.readlines():
            path, target = line.strip().split()
            item = (path, class_to_idx[target])
            images.append(item)
    return images


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class TextLine(data.Dataset):
    def __init__(self, root, listfile, transform=None, target_transform=None, loader=default_loader, extensions=IMG_EXTENSIONS):
        classes, class_to_idx = self._find_classes(listfile)
        samples = make_dataset(listfile, class_to_idx, extensions)
        if not samples:
            raise(RuntimeError("Found 0 samples in listfile: {}".format(listfile)))

        self.root = root        
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.imgs = self.samples

        self.transform = transform
        self.target_transform = target_transform
    
    def _find_classes(self, listfile):
        with open(listfile) as f:
            classes = set( [line.strip().split()[1] for line in f.readlines()] )
        classes = list(classes)
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(os.path.join(self.root, path))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
