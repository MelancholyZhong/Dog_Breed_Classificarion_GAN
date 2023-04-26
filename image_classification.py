# ================================== #
#        Created by Hui Hu           #
#         Classify Images            #
# ================================== #

import os

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models



# Draw negative log likelihood loss chart
def drawNLLL(train_losses, train_counter, test_losses, test_counter, image_name):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    fig.savefig(f"results/images/{image_name}.png")


# Train model for one epoch
def train_network(model, train_loader, train_losses, train_counter, epoch, batch_size=64, log_interval=10, learning_rate=0.01, momentum = 0.5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Define the loss function and optimizer
    loss_fn = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Define lists to store the training loss for this epoch
    epoch_train_losses = []
    epoch_train_counter = []

    # Train the model for one epoch
    model.train()
    for batch_idx, (d, t) in enumerate(train_loader):
        data, target = d.to(device), t.to(device)
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    min(batch_idx * batch_size, len(train_loader.dataset)),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_losses.append(loss.item())
            train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))
            epoch_train_losses.append(loss.item())
            epoch_train_counter.append((batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))

    return epoch_train_losses, epoch_train_counter


# Test model for one epoch
def test(model, test_loader, test_losses):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    test_loss = 0
    correct = 0
    loss_fn = nn.NLLLoss(reduction="sum").to(device)  # loss function

    # Evaluate the model on the test sets
    with torch.no_grad():
        for d, t in test_loader:
            data, target = d.to(device), t.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)

    test_losses.append(test_loss)
    print(
        "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )



# Training model
def model_training():
    # configure
    epochs = 7
    train_batch_size = 128
    names = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create results directory to save images if not exists
    directory1 = "results/models"
    os.makedirs(directory1, exist_ok=True)
    directory2 = "results/images"
    os.makedirs(directory2, exist_ok=True)

    # Set random seed for repeatability
    torch.manual_seed(42)
    torch.backends.cudnn.enabled = True if torch.cuda.is_available() else False # false -> turn off CUDA
    
    # Load MNIST data set
    train_set = torchvision.datasets.ImageFolder(
        "data/train/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4354,), (0.2409,)),
            ]
        ),
    )
    test_set = torchvision.datasets.ImageFolder(
        "data/test/",
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4354,), (0.2409,)),
            ]
        ),
    )

    # Define the training and test data loaders with the specified batch size
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1000,
        shuffle=True,
    )

    # Show six example dog breeds
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(8, 6))
    for i in range(6):
        img, label = train_set[i * 110]
        row = i // 3
        col = i % 3
        axs[row, col].imshow(cv2.UMat(img[0].numpy()).get())
        axs[row, col].set_title(f"Label: {label}")
    fig.savefig("results/dogs_example.png")


    # create models
    model_lst = [None, None, None, None, None]

    # # ResNet18
    model_lst[0] = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    model_lst[0].fc = nn.Sequential(nn.Linear(in_features=512, out_features=120, bias=True), nn.LogSoftmax(dim=1))

    # # ResNet34
    model_lst[1] = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
    model_lst[1].fc = nn.Sequential(nn.Linear(in_features=512, out_features=120, bias=True), nn.LogSoftmax(dim=1))

    # ResNet50
    model_lst[2] = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(device)
    model_lst[2].fc = nn.Sequential(nn.Linear(in_features=2048, out_features=120, bias=True), nn.LogSoftmax(dim=1))

    # # ResNet101
    model_lst[3] = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2).to(device)
    model_lst[3].fc = nn.Sequential(nn.Linear(in_features=2048, out_features=120, bias=True), nn.LogSoftmax(dim=1))

    # # ResNet152
    model_lst[4] = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).to(device)
    model_lst[4].fc = nn.Sequential(nn.Linear(in_features=2048, out_features=120, bias=True), nn.LogSoftmax(dim=1))
    
    for i in range(len(model_lst)):
      model = model_lst[i]

      print(f"\n\n\n{names[i]}\n\n\n")

      train_losses = []
      train_counter = []
      test_losses = []
      test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

      # Loop over the epochs
      test(model, test_loader, test_losses)
      for epoch in range(1, epochs + 1):
          epoch_train_losses, epoch_train_counter = train_network(model, train_loader, train_losses, train_counter, epoch, batch_size=train_batch_size, log_interval=10, learning_rate=0.08, momentum=0.5)
          test(model, test_loader, test_losses)
          epoch_test_losses = test_losses[: epoch + 1]
          if epoch > 1:
              epoch_test_losses[0] = 0
          epoch_test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]
          # drawNLLL(epoch_train_losses, epoch_train_counter, epoch_test_losses, epoch_test_counter, f"digit_epoch_{epoch}")

      drawNLLL(train_losses, train_counter, test_losses, test_counter, names[i])

      # Save the model to a file
      torch.save(model.state_dict(), "results/models/" + names[i] + ".pth")

if __name__ == "__main__":
    model_training()