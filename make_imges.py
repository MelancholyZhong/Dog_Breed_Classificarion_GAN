# CS5330 
# helper function that loads a model, generates noises and shows the generated images
# Author: Yao Zhong, zhong.yao@northeastern.edu

# import statements

from __future__ import print_function
import sys
import argparse
import os
import random
import torch
from torch import nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils


from breed_CGAN_v4 import GeneratorModel


# main function
def main(argv):
    
    random_seed = 51
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    
    #load pre-trained
    generator = torch.load('./trained_models/v4_50_dogs_c_gan_G_model.pth', map_location=torch.device('cpu'))

    # Show the final generated examples
    breed = 81
    figure=plt.figure(figsize=(6,2)) #8x6 inches window
    cols, rows = 3, 1
    with torch.no_grad():
        noise = torch.randn(3,100).to(device)
        labels = (torch.tensor([1,1,1], dtype=torch.int32)*breed).to(device)
        generated_data = generator(noise, labels).view(-1, 3, 64, 64)
        for idx, x in enumerate(generated_data):
            figure.add_subplot(rows, cols, idx+1)
            plt.axis("off")
            plt.title(labels[idx].item())
            plt.imshow(np.transpose(x.detach()*0.7+0.6, (1,2,0)))
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)