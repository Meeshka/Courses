import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np


# create training dataset
def generate_dataset(test):
    data_set = datasets.FashionMNIST(
        root="data",
        train=test,
        download=True,
        transform=ToTensor()
    )
    return data_set


# create a validate sample from the training dataset
def create_sample(data_set):
    indices = list(range(len(data_set)))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * len(data_set)))
    sample = SubsetRandomSampler(indices[:split])
    validate_sample = SubsetRandomSampler(indices[split:])
    return sample, validate_sample


if __name__ == "__main__":
    print("Generating training...")
    training_dataset = generate_dataset(test=False)
    print("Generating testing...")
    testing_dataset = generate_dataset(test=True)
    print("Creating sample training...")
    train_sample, validate_train = create_sample(training_dataset)
    print("Creating sample testing...")
    test_sample, validate_test = create_sample(testing_dataset)

    # data loader
    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(training_dataset, sampler=train_sample, batch_size=64)
    valid_loader = torch.utils.data.DataLoader(training_dataset, sampler=validate_train, batch_size=64)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=True)

    data_iter = iter(train_loader)
    images, labels = data_iter.__next__()

    print("Printing out...")
    figure = plt.figure(figsize=(15,15))
    for i in np.arange(20):
        ax = figure.add_subplot(4, int(20 / 4), i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[i]), cmap='gray')
        ax.set_title(labels[i].item())
        figure.tight_layout()
    plt.show()