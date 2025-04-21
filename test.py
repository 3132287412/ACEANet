import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, exists
# pytorch libs
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from model import Trans

def test(opt):
    '''  ------testing pipeline------  '''
    assert exists(opt.model_path), "model not found"
    os.makedirs(opt.sample_dir, exist_ok=True)  # Ensure sample directory exists
    is_cuda = torch.cuda.is_available()
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    # model architecture
    model = Trans.MyNet()
    ## load weights
    model.load_state_dict(torch.load(opt.model_path, map_location='cuda' if is_cuda else 'cpu'))

    if is_cuda:
        model.cuda()
    print("Loaded model from %s" % (opt.model_path))

    ## data pipeline
    img_width, img_height, channels = 256, 256, 3
    transforms_ = [
        transforms.Resize((img_height, img_width), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(transforms_)

    ## testing loop
    times = []
    test_files = sorted(glob(join(opt.data_dir, "*.*")))
    for path in test_files:
        inp_img = transform(Image.open(path))
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        
        # generate enhanced image
        s = time.time()
        gen_img, ll = model(inp_img)
        times.append(time.time() - s)
        
        # Save the generated image to the samples directory
        save_image(gen_img[0], join(opt.sample_dir, os.path.basename(path)), normalize=True)
        print("Tested: %s" % path)

if __name__ == '__main__':
    ## options
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./scenes-130", help="path of test images")
    parser.add_argument("--sample_dir", type=str, default="./data/fei/", help="path to save samples")
    parser.add_argument("--model_path", type=str, default="checkpoints/generator_300.pth")  # 240轮  300
    opt = parser.parse_args()
    test(opt)