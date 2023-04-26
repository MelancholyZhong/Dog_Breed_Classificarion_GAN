# CS5330 
# Author: Yao Zhong, zhong.yao@northeastern.edu
# version 3 of the Breed Generator, improved the model of discriminator
# second edition of using deep convolution network
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
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


#load and transform the dog images from the specified dataset
def loadData(image_path):
    # DataLoader for the dog images
    true_dogs = torchvision.datasets.ImageFolder( image_path,
                    transform = transforms.Compose( [transforms.Resize(64),
                                                    transforms.CenterCrop(64),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) #mean being 0.5
                                                    ] ) )

    return true_dogs

#Helper function that plots the first 9 images from the dog images
def showExamples(training_data):
    size = 9
    figure=plt.figure(figsize=(8,6)) #8x6 inches window
    cols, rows = 3, (size+2)//3
    for i in range(cols*rows):
        img, label = training_data[i]
        figure.add_subplot(rows, cols, i+1)
        plt.title(label)
        plt.axis("off")
        # The tensor is mean 0.5 and [3,32,32], to show it, need to time 255 and add 128 then change demension
        plt.imshow(np.transpose(img+1, (1,2,0)))
    plt.show()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# The structue of the generator model
class GeneratorModel(nn.Module):
    def __init__(self):
        super(GeneratorModel, self).__init__()
        input_dim =100 + 20 #additional 120 dimentions for the embedding of the dogs
        ngf = 64
        #embeddings for the 120 breeds
        self.label_embedding = nn.Embedding(120,20)
        # This is from the tutorial architecture.
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

            
        
    # computes a forward pass for the network
    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c],1) #append the embedings after the noise (100+20)
        x = x.view(-1, 120, 1,1)
        output = self.main(x)
        return output
    
# The structue of the discriminator model
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()
        # 64*64*3 is 12,288, which is the size of the dog images
        # output dim is the possibility of this image is a dog or not
        ndf = 64
        #embeddings for the 120 dog
        self.label_embedding = nn.Embedding(120,20)
        # This is from the tutorial architecture.
        self.hidden_1 = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 100, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden_2  = nn.Sequential(
            nn.Linear(120, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    # computes a forward pass for the network
    def forward(self, x, labels):
        c = self.label_embedding(labels)
        output = x.view(-1,3,64,64)
        output = self.hidden_1(output)
        output = output.view(-1, 100)
        output = torch.cat([output,c], 1)
        output = self.hidden_2(output)
        return output

# The function that trians the network and plots the result of the trianing and testing
def train_GAN(dataloader, G, D, loss, G_optimizer, D_optimizer, epochs, device):
    #holder for the result of each epoch
    G_loss = []
    D_loss = []
    counter = []

    #call the train_loop and test_loop for each epoch
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(dataloader, G, D, loss, G_optimizer, D_optimizer, G_loss, D_loss, counter, t, device)
    print("Done!")

    # Plot the training and testing result
    fig = plt.figure()
    plt.plot(counter, G_loss, color='blue')
    plt.plot(counter, D_loss, color='red')
    plt.legend(['Generater Loss', 'Discriminator Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

# The train_loop for one epoch, the statistics are saved in losses
def train_loop(dataloader, G, D, loss, G_optimizer, D_optimizer, G_loss, D_loss, counter, epochIdx, device):
    # size of the whole set, used in counter
    size = len(dataloader.dataset)
    for batchIdx, (data, target) in enumerate(dataloader):
        batch_size = len(data)
        # Generate noise and move it the device
        noise = torch.randn(batch_size,100).to(device) #the noise size is 100
        fake_labels = torch.randint(0,120,(batch_size, )).to(device) #the labels are [0,120)
        #forward
        generated_data = G(noise, fake_labels).to(device) #batch_size x 100

        true_data = data.view(batch_size, -1).to(device) #batch_size x 100
        digit_labels = target.to(device) # batch_size
        true_labels = torch.ones(batch_size).to(device)
        
        # Clear optimizer gradients
        D_optimizer.zero_grad()
        # Forward pass with true data as input
        D_true_output = D(true_data, digit_labels).to(device).view(batch_size)
        # Compute Loss
        D_true_loss = loss(D_true_output, true_labels)
        # Forward pass with generated data as input
        D_generated_output = D(generated_data.detach().view(batch_size,-1), fake_labels).to(device).view(batch_size)
        # Compute Loss 
        D_generated_loss = loss(
            D_generated_output, torch.zeros(batch_size).to(device)
        )
        # Average the loss
        D_average_loss = (
            D_true_loss + D_generated_loss
        ) / 2
               
        # Backpropagate the losses for Discriminator model      
        D_average_loss.backward()
        D_optimizer.step()

        # Clear optimizer gradients
        G_optimizer.zero_grad()
        
        # It's a choice to generate the data again
        generated_data = G(noise, fake_labels).to(device) # batch_size X 100
        # Forward pass with the generated data
        D_generated_output = D(generated_data.view(batch_size,-1), fake_labels).to(device).view(batch_size)
        # Compute loss
        generator_loss = loss(D_generated_output, true_labels)
        # Backpropagate losses for Generator model.
        generator_loss.backward()
        G_optimizer.step()
        
     
        # log in the terminal for each 10 batches
        if batchIdx % 10 == 0:
            current = (batchIdx+1)*len(data)
            print(f"D_loss: {D_average_loss.data.item():>7f}  G_loss: {generator_loss.data.item():>7f}  [{current:>5d}/{size:>5d}]")
            D_loss.append(D_average_loss.data.item())
            G_loss.append(generator_loss.data.item())
            counter.append(batchIdx*len(data)+epochIdx*size)
        
        if ((batchIdx + 1)% 500 == 0 and (epochIdx + 1)%2 == 0):
            
            with torch.no_grad():
                noise = torch.randn(batch_size,100).to(device)
                fake_labels = torch.randint(0,120,(batch_size, )).to(device)
                generated_data = G(noise,fake_labels)
                for x in generated_data:
                    plt.imshow(np.transpose(x.detach().cpu()+1, (1,2,0)))
                    plt.show()
                    break

# main function
def main(argv):
    
    random_seed = 47
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.cuda.empty_cache()

    # Set the settings for the training
    learning_rate = 2e-4
    batch_size = 32
    epochs = 200
    beta1 = 0.5

    #load traing data
    data = loadData('./data/crop_dogs')
    # Uncomment to show the first six images.
    showExamples(data)

    # create dataloaders
    dataloader = DataLoader(data, batch_size, shuffle=True)

    # Create the discriminator and generator
    generator = GeneratorModel()
    discriminator = DiscriminatorModel()

    
    # #load pre-trained
    # generator = torch.load('./dogs_c_gan_G_model.pth')
    # discriminator = torch.load('./dogs_c_gan_D_model.pth')

    # move them to device
    generator.to(device)
    discriminator.to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # use binary cross-entropy loss
    loss = nn.BCELoss()
    # use Adam optimizer for both model
    generator_optimizer = torch.optim.Adam(generator.parameters(), learning_rate,betas=(beta1, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), learning_rate,betas=(beta1, 0.999))

    train_GAN(dataloader, generator, discriminator, loss, generator_optimizer, discriminator_optimizer, epochs, device)

    # Show the final generated examples
    figure=plt.figure(figsize=(8,6)) #8x6 inches window
    cols, rows = 3, 4
    with torch.no_grad():
        noise = torch.randn(10,100).to(device)
        labels = torch.tensor([0,1,2,3,4,5,6,7,8,9], dtype=torch.int32).to(device)
        generated_data = generator(noise, labels)
        for idx, x in enumerate(generated_data):
            figure.add_subplot(rows, cols, idx+1)
            plt.axis("off")
            plt.title(labels[idx].item())
            plt.imshow(np.transpose(x.detach().cpu()+1, (1,2,0)))
    plt.show()

    # Save the trained models
    torch.save(discriminator, 'dogs_c_gan_D_model.pth')
    torch.save(generator, 'dogs_c_gan_G_model.pth')

    return

if __name__ == "__main__":
    main(sys.argv)