
import os
import random

import numpy as np
import cv2
from glob import glob
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def read_image(imagefile, grayscale=False):
    if grayscale == True:
        image = cv2.imread(imagefile)
        #image = np.expand_dims(image, -1)
    else:
        image = cv2.imread(imagefile)
    return image

def save_image(image, mask, path, binary=True):
    image = np.array(image)
    if binary == True:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(path[0], image)
    cv2.imwrite(path[1], mask)

def concat_images(images, rows, cols):
    _, h, w, _ = images.shape
    images = images.reshape((rows, cols, h, w, 3))
    images = images.transpose(0, 2, 1, 3, 4)
    images = images.reshape((rows * h, cols * w, 3))
    return images

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

def subtract(image):
    image = image / 255
    return image

def resize(image, size):
    size = check_size(size)
    image = cv2.resize(image, size)
    return image

def center_crop(image, mask, crop_size, size):
    h, w, _ = image.shape
    crop_size = check_size(crop_size)
    top = (h - crop_size[0]) // 2
    left = (w - crop_size[1]) // 2
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def random_crop(image, mask, crop_size, size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    image = image[top:bottom, left:right, :]
    mask = mask[top:bottom, left:right, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def horizontal_flip(image, mask, size):
    image = image[:, ::-1, :]
    mask = mask[:, ::-1, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def vertical_flip(image, mask, size):
    image = image[::-1, :, :]
    mask = mask[::-1, :, :]

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def scale_augmentation(image, mask, scale_range, crop_size, size):
    scale_size = np.random.randint(*scale_range)
    image = cv2.resize(image, (scale_size, scale_size))
    mask = cv2.resize(mask, (scale_size, scale_size))
    image, mask = random_crop(image, mask, crop_size, size)
    return image, mask

def random_rotation(image, mask, size, angle_range=(0, 90)):
    h1, w1, _ = image.shape
    h2, w2, _ = mask.shape

    angle = np.random.randint(*angle_range)
    image = rotate(image, angle)
    image = resize(image, (h1, w1))

    mask = rotate(mask, angle)
    mask = resize(mask, (h2, w2))

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def cutout(image_origin, mask_origin, mask_size, mask_value='mean'):
    image = np.copy(image_origin)
    mask = np.copy(mask_origin)

    if mask_value == 'mean':
        mask_value = image.mean()
    elif mask_value == 'random':
        mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    image[top:bottom, left:right, :].fill(mask_value)
    mask[top:bottom, left:right, :].fill(0)

    image = resize(image, size)
    mask = resize(mask, size)

    return image, mask

def brightness_augment(img, mask, factor=0.5):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #convert to hsv
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform()) #scale channel V uniformly
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 #reset out of range values
    rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

    image = resize(rgb, size)
    mask = resize(mask, size)

    return image, mask

def rgb_to_grayscale(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = [img, img, img]
    img = np.transpose(img, (1, 2, 0))

    image = resize(img, size)
    mask = resize(mask, size)
    return image, mask

def create_dir(name):
    try:
        os.mkdir(name)
    except:
        pass

if __name__ == '__main__':
    ### Image Augmentation
    size = (256, 256)
    crop_size = (300, 300)

    path = "../../../ml_dataset/"
    dataset_name = "kvasir_segmentation_dataset"
    full_path = os.path.join(path, dataset_name)

    new_path = "new_data/"
    create_dir(new_path)
    new_full_path = os.path.join(new_path, dataset_name)

    train_path = os.path.join(new_full_path, "train")
    valid_path = os.path.join(new_full_path, "valid")
    test_path = os.path.join(new_full_path, "test")

    if not os.path.exists(new_full_path):
        os.mkdir(new_full_path)
        for path in [train_path, valid_path, test_path]:
            os.mkdir(path)
            os.mkdir(os.path.join(path, "images"))
            os.mkdir(os.path.join(path, "masks"))

    images = glob(os.path.join(full_path, "images/", "*"))
    masks = glob(os.path.join(full_path, "masks/", "*"))

    images.sort()
    masks.sort()

    len_ids = len(images)
    train_size = int((80/100)*len_ids)
    valid_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for validation
    test_size = int((10/100)*len_ids)		## Here 10 is the percent of images used for testing

    train_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
    train_masks, test_masks = train_test_split(masks, test_size=test_size, random_state=42)

    train_images, valid_images = train_test_split(train_images, test_size=test_size, random_state=42)
    train_masks, valid_masks = train_test_split(train_masks, test_size=test_size, random_state=42)

    print("Total Size: ", len_ids)
    print("Training Size: ", train_size)
    print("Validation Size: ", valid_size)
    print("Testing Size: ", test_size)

    ## Validation images and masks
    for idx, p in tqdm(enumerate(test_images), total=len(test_images)):
        ## Path
        name = p.split("/")[-1].split(".")[0]
        image_path = test_images[idx]
        mask_path = test_masks[idx]

        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = read_image(image_path)
            mask = read_image(mask_path, grayscale=True)

            new_image_path = os.path.join(new_full_path, "test", "images/")
            new_mask_path = os.path.join(new_full_path, "test", "masks/")

            image = resize(image, size)
            mask = resize(mask, size)

            img_path = new_image_path + str(name) + ".jpg"
            mask_path = new_mask_path + str(name) + ".jpg"
            tmp_path = [img_path, mask_path]
            save_image(image, mask, tmp_path)

    ## Testing images and masks
    for idx, p in tqdm(enumerate(valid_images), total=len(valid_images)):
        ## Path
        name = p.split("/")[-1].split(".")[0]
        image_path = valid_images[idx]
        mask_path = valid_masks[idx]

        if os.path.exists(image_path) and os.path.exists(mask_path):
            image = read_image(image_path)
            mask = read_image(mask_path, grayscale=True)

            new_image_path = os.path.join(new_full_path, "valid", "images/")
            new_mask_path = os.path.join(new_full_path, "valid", "masks/")

            image = resize(image, size)
            mask = resize(mask, size)

            img_path = new_image_path + str(name) + ".jpg"
            mask_path = new_mask_path + str(name) + ".jpg"
            tmp_path = [img_path, mask_path]
            save_image(image, mask, tmp_path)

    ## Training images and masks
    for idx, p in tqdm(enumerate(train_images), total=len(train_images)):
        ## Path
        name = p.split("/")[-1].split(".")[0]
        image_path = train_images[idx]
        mask_path = train_masks[idx]

        if os.path.exists(image_path) and os.path.exists(image_path):
            image = read_image(image_path)
            mask = read_image(mask_path, grayscale=True)

            ## Augment
            image1, mask1 = center_crop(image, mask, crop_size, size)
            image2, mask2 = random_crop(image, mask, crop_size, size)
            image3, mask3 = horizontal_flip(image, mask, size)
            image4, mask4 = vertical_flip(image, mask, size)
            image5, mask5 = scale_augmentation(image, mask, (512, 768), crop_size, size)
            image6, mask6 = random_rotation(image, mask, size)
            image7, mask7 = cutout(image, mask, 256)
            ## Extra Cropping
            image8, mask8 = random_crop(image, mask, crop_size, size)
            image9, mask9 = random_crop(image, mask, crop_size, size)
            ## Extra Scale Augmentation
            image10, mask10 = scale_augmentation(image, mask, (540, 820), crop_size, size)
            image11, mask11 = scale_augmentation(image, mask, (720, 1024), crop_size, size)
            ## Extra Rotation
            image12, mask12 = random_rotation(image, mask, size)
            image13, mask13 = random_rotation(image, mask, size)
            ## Brightness
            image14, mask14 = brightness_augment(image, mask, factor=0.3)
            image15, mask15 = brightness_augment(image, mask, factor=0.6)
            image16, mask16 = brightness_augment(image, mask, factor=0.9)
            ## More Rotation
            image17, mask17 = random_rotation(image, mask, size)
            image18, mask18 = random_rotation(image, mask, size)
            ## More Random Crop
            image19, mask19 = random_crop(image, mask, crop_size, size)
            image20, mask20 = random_crop(image, mask, crop_size, size)
            ## More Cutout
            image21, mask21 = cutout(image, mask, 256)
            image22, mask22 = cutout(image, mask, 256)
            ## Grayscale
            image23, mask23 = rgb_to_grayscale(image, mask)
            image24, mask24 = rgb_to_grayscale(image1, mask1)
            image25, mask25 = rgb_to_grayscale(image2, mask2)
            image26, mask26 = rgb_to_grayscale(image3, mask3)
            image27, mask27 = rgb_to_grayscale(image4, mask4)
            image28, mask28 = rgb_to_grayscale(image5, mask5)
            image29, mask29 = rgb_to_grayscale(image15, mask15)
            image30, mask30 = rgb_to_grayscale(image16, mask16)

            ## Original image and mask
            image = resize(image, size)
            mask = resize(mask, size)

            ## All images and masks
            all_images = [image, image1, image2, image3, image4, image5, image6, image7,
                image8, image9, image10, image11, image12, image13, image14, image15, image16,
                image17, image18, image19, image20, image21, image22,
                image23,image24, image25, image26, image27, image28, image29, image30
                ]
            all_masks  = [mask, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8,
                mask9, mask10, mask11, mask12, mask13, mask14, mask15, mask16,
                mask17, mask18, mask19, mask20, mask21, mask22,
                mask23, mask24, mask25, mask26, mask27, mask28, mask29, mask30
                ]

            ## Save the images and masks
            new_image_path = os.path.join(new_full_path, "train", "images/")
            new_mask_path = os.path.join(new_full_path, "train", "masks/")

            for j in range(len(all_images)):
                img_path = new_image_path + str(name) + "_" + str(j) + ".jpg"
                msk_path = new_mask_path + str(name) + "_" + str(j) + ".jpg"

                img = all_images[j]
                msk = all_masks[j]
                path = [img_path, msk_path]

                save_image(img, msk, path)
