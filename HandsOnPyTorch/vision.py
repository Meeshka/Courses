import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import read_image
from PIL import Image
import torch.nn as nn
from scipy.ndimage import gaussian_filter


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

def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + (1 if with_orig else 0)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [imgs[row_idx][0]] + row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    plt.show()

def show_images(imgs):
    show(imgs)

def     show_PIL_transform():
    orig_img = Image.open(Path('images') / 'pottery1.jpeg')
    blurred_image = gaussian_filter(orig_img, sigma=10)
    plot([orig_img, blurred_image], True, row_title=['Blurred'], cmap='gray')


def show_random_crop(imgs):
    transforms = torch.nn.Sequential(
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.3),
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pottery1 = imgs[0].to(device)
    pottery2 = imgs[1].to(device)

    transformed_pottery1 = transforms(pottery1)
    transformed_pottery2 = transforms(pottery2)
    show([transformed_pottery1, transformed_pottery2])

def get_images():
    pottery1 = read_image(str(Path('images')/'pottery1.jpeg'))
    pottery2 = read_image(str(Path('images')/'pottery3.png'))
    return [pottery1, pottery2]

def show_color_jitter(img):
    jitter = T.ColorJitter(brightness=.5, hue=.3)
    jitted_imgs = [jitter(img) for _ in range(4)]
    jitted_imgs = [np.transpose(jitted_img, (1, 2, 0)) for jitted_img in jitted_imgs]
    plot(jitted_imgs)

def show_randow_perspective(img):
    perspective_transformer = T.RandomPerspective(distortion_scale=0.6, p=1.0)
    perspective_imgs = [perspective_transformer(img) for _ in range(4)]
    perspective_imgs = [np.transpose(perspective_img, (1, 2, 0)) for perspective_img in perspective_imgs]
    plot(perspective_imgs)

def show_center_crops(img):
    img_size = img.size()
    center_crops = [T.CenterCrop(size=size)(img) for size in (30, 50, 100, min(img_size))]
    center_crops = [np.transpose(center_crop, (1, 2, 0)) for center_crop in center_crops]

    plot(center_crops)

if __name__ == "__main__":

    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(1)

    imgs = get_images()

    #show_list()

    #show_images(imgs)

    #show_PIL_transform()

    #show_random_crop(imgs)

    #show_color_jitter(imgs[0])

    #show_randow_perspective(imgs[0])

    show_center_crops(imgs[0])