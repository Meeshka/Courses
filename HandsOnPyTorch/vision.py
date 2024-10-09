import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'
torch.manual_seed(1)

def show(imgs):
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = T.ToPILImage()(img.to('cpu'))
        axs[0,i].imshow(np.asarray(img))
        axs[0,i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()


def show_list():
    images_path = Path('images')

    # List all files matching a specific mask (e.g., all .jpeg and .png files)
    jpeg_files = list(images_path.glob('*.jpeg'))
    png_files = list(images_path.glob('*.png'))

    # Combine both lists
    all_images = jpeg_files + png_files

    # Print the list of files
    for img in all_images:
        print(img.name)


if __name__ == "__main__":

    #show_list()

    pottery1 = read_image(str(Path('images')/'pottery1.jpeg'))
    pottery2 = read_image(str(Path('images')/'pottery3.png'))
    show([pottery1,pottery2])
