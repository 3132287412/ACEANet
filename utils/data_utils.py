import os
import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GetTrainingPairs(Dataset):
    """ Common data pipeline to organize and generate
         training pairs for various datasets
    """
    def __init__(self, root, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.filesA, self.filesB = self.get_file_paths(root)
        self.len = min(len(self.filesA), len(self.filesB))

    def __getitem__(self, index):
        img_A = Image.open(self.filesA[index % self.len])
        img_B = Image.open(self.filesB[index % self.len])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return self.len

    def get_file_paths(self, root):
        filesA, filesB = [], []
        filesA += sorted(glob.glob(os.path.join(root, 'trainA') + "/*.*"))
        filesB += sorted(glob.glob(os.path.join(root, 'trainB') + "/*.*"))

        print("Train_dataset", len(filesA))
        return filesA, filesB
class GetValImage(Dataset):
    """ Common data pipeline to organize and generate
         vaditaion samples for various datasets
    """
    def __init__(self, root, transforms_=None, sub_dir='validation'):
        self.transform = transforms.Compose(transforms_)
        self.files = self.get_file_paths(root)
        self.len = len(self.files)

    def __getitem__(self, index):
        img_val = Image.open(self.files[index % self.len])
        img_val = self.transform(img_val)
        return {"val": img_val}

    def __len__(self):
        return self.len

    def get_file_paths(self, root):
        files = []
        files += sorted(glob.glob(os.path.join(root, 'trainB') + "/*.*"))
        #print("validation", len(files))
        return files

