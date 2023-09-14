import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
from pathlib import Path  

from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset

def make_dataset(root: str, sub1: str, sub2: str) -> list:
    """Reads a directory with data.
    Returns a dataset as a list of tuples of paired image paths: (rgb_path, gt_path)
    """
    dataset = []

    # Get all the filenames from RGB folder
    vis_fnames = sorted(os.listdir(os.path.join(root, sub1)))
    ir_names = sorted(os.listdir(os.path.join(root, sub2)))
    
    # Compare file names from GT folder to file names from RGB:
    for ir_fname in ir_names:

            if ir_fname in vis_fnames:
                # if we have a match - create pair of full path to the corresponding images
                vis_path = os.path.join(root, sub1, ir_fname)
                ir_path = os.path.join(root, sub2, ir_fname)

                item = (vis_path, ir_path)
                # append to the list dataset
                dataset.append(item)
            else:
                continue

    return dataset


class CustomVisionDataset_train(VisionDataset):
    def __init__(self,
                 root,
                 subfolder1,
                 subfolder2,
                 loader=default_loader):
        super().__init__(root)

        # Prepare dataset
        samples = make_dataset(self.root, subfolder1, subfolder2)

        self.loader = loader
        self.samples = samples
        # list of RGB images
        self.vis_samples = [s[1] for s in samples]
        # list of GT images
        self.ir_samples = [s[1] for s in samples]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        vis_path, ir_path = self.samples[index]
        
        # import each image using loader (by default it's PIL)
        vis_sample = self.loader(vis_path)
        ir_sample = self.loader(ir_path)
        
        # vis_sample = vis_sample.convert('L')
        ir_sample = ir_sample.convert('L')
        ir_sample = np.stack((ir_sample,)*3, axis=-1)
        ir_sample = Image.fromarray(ir_sample.astype('uint8'))

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        vis_sample = self.transform(vis_sample)
        ir_sample = self.transform(ir_sample)
        
        #get name for sure
        vis_name = Path(vis_path).name
        ir_name = Path(ir_path).name

        # now we return the right imported pair of images (tensors)
        return vis_sample, ir_sample, vis_name, ir_name

    def __len__(self):
        return len(self.samples)

class CustomVisionDataset_test(VisionDataset):
    def __init__(self,
                 root,
                 subfolder1,
                 subfolder2,
                 loader=default_loader):
        super().__init__(root)

        # Prepare dataset
        samples = make_dataset(self.root, subfolder1, subfolder2)

        self.loader = loader
        self.samples = samples
        # list of RGB images
        self.vis_samples = [s[1] for s in samples]
        # list of GT images
        self.ir_samples = [s[1] for s in samples]
        
        self.transform = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])


    def __getitem__(self, index):
        """Returns a data sample from our dataset.
        """
        # getting our paths to images
        vis_path, ir_path = self.samples[index]
        
        # import each image using loader (by default it's PIL)
        vis_sample = self.loader(vis_path)
        ir_sample = self.loader(ir_path)
        
        # vis_sample = vis_sample.convert('L')
        ir_sample = ir_sample.convert('L')
        ir_sample = np.stack((ir_sample,)*3, axis=-1)
        ir_sample = Image.fromarray(ir_sample.astype('uint8'))

        # here goes tranforms if needed
        # maybe we need different tranforms for each type of image
        vis_sample = self.transform(vis_sample)
        ir_sample = self.transform(ir_sample)
        
        #get name for sure
        vis_name = Path(vis_path).name
        ir_name = Path(ir_path).name

        # now we return the right imported pair of images (tensors)
        return vis_sample, ir_sample, vis_name, ir_name

    def __len__(self):
        return len(self.samples)


def mkdir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)