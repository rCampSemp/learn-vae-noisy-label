import torch
from torchvision.datasets import MNIST 
import os
import cv2
import numpy as np
from tifffile import imsave


def divide_cases(trainset, test_set):
    """Divides mnist datasets into train val and test split datasets.

    Args:
        trainset (torchvision.datasets.MNIST): MNIST train dataset train
        testset (torchvision.datasets.MNIST): MNIST test dataset

    Returns:
        train_set (Subset object): subset of MNIST dataset used for training
        val_set (Subset object): subset of MNIST dataset used for validation
        test_set (Subset object): subset of MNIST dataset used for testing
    """

    train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000])
    return train_set, val_set, test_set

def gauss_transform(img, mean=2, std=5, weight=0.5):
    """Function to transform MNIST img to img with guassian noise added and normalised.

    Args:
        img (numpy array): input img array
        mean (int, optional): mean of guassian distribution. Defaults to 2.
        std (int, optional): standard deviation of guassian distrbution. Defaults to 5.
        weight (float, optional): weight of blending of the input img with gaussian noise. Defaults to 0.5.

    Returns:
        img (numpy array): _description_
    """
    img = img.copy()
    # generate gaussian noise in same shape as input img
    gaussian = np.uint8(np.random.normal(mean, std, (img.shape[0],img.shape[1])))
    
    # weighted addition of gaussian noise with img, 0 - 255 => 0 - 1 range
    img = cv2.addWeighted(img, 1.0, gaussian, weight, 0) / 255
    return img

def main(data, save_path, split):
    """main process

    Args:
        data (Subset object): subset of MNIST dataset
        save_path (str): file path to save processed MNIST data
        split (str): subfolder to save data to, train/validate/test
    """
    # file path depending on data type to save data to
    save_path = save_path + split
    os.makedirs(save_path, exist_ok=True)
    
    # initialize subfolders to save data to
    save_gauss = save_path + '/Gaussian'
    save_GT = save_path + '/GT'
    save_over = save_path + '/Over'
    save_under = save_path + '/Under'
    save_wrong = save_path + '/Wrong'

    os.makedirs(save_gauss, exist_ok=True)
    os.makedirs(save_GT, exist_ok=True)
    os.makedirs(save_over, exist_ok=True)
    os.makedirs(save_under, exist_ok=True)
    os.makedirs(save_wrong, exist_ok=True)

    # kernel for morphological operations
    kernel = np.ones((2, 2), np.uint8)
    
    # iterate over given subset with labels not needed
    for idx, (img, _) in enumerate(data):
        # opencv compatible type
        img = np.asarray(img, dtype=np.uint8)

        # get gaussian img transformation
        gauss_img = gauss_transform(img)
        # make gauss img 3 channels for compatibility with customdataset_punet
        gauss_img = gauss_img.reshape((28, 28, 1)).repeat(3, -1) # repeat the last (-1) dimension three times

        # under and over annotators
        under_img = cv2.erode(img, kernel, iterations=2)
        over_img = cv2.dilate(img, kernel, iterations=2)

        ## wrong segmentation production
        rows, cols = img.shape

        # rotate img 135 degrees
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),135,1) 
        wrong_img = cv2.warpAffine(img,M,(cols,rows))

        # morpological operations to produce wrong segmentation
        wrong_img = cv2.morphologyEx(wrong_img, cv2.MORPH_TOPHAT, kernel)
        wrong_img = cv2.dilate(wrong_img, kernel, iterations=4)
        wrong_img = cv2.morphologyEx(wrong_img, cv2.MORPH_TOPHAT, kernel)
        wrong_img = cv2.dilate(wrong_img, kernel, iterations=4)

        # save name for each subfolder
        save_gauss_file = save_gauss + '/' + str(idx) + '_Gaussian.tif'
        save_under_file = save_under + '/' + str(idx) + '_Under.tif'
        save_over_file = save_over + '/' + str(idx) + '_Over.tif'
        save_wrong_file = save_wrong + '/' + str(idx) + '_Wrong.tif'
        save_GT_file = save_GT + '/' + str(idx) + '_GT.tif'

        # threshold images to make binary mask
        (_, wrong_img) = cv2.threshold(wrong_img, 50, 1,
            cv2.THRESH_BINARY)
        (_, under_img) = cv2.threshold(under_img, 0, 1,
            cv2.THRESH_BINARY)
        (_, over_img) = cv2.threshold(over_img, 0, 1,
            cv2.THRESH_BINARY)
        (_, GT_img) = cv2.threshold(img, 0, 1,
            cv2.THRESH_BINARY)

        # save masks to subfolders
        imsave(save_gauss_file, gauss_img)
        imsave(save_GT_file, GT_img)
        imsave(save_over_file, over_img)
        imsave(save_under_file, under_img)
        imsave(save_wrong_file, wrong_img)

if __name__ == '__main__':
    # get MNIST datasets
    trainset = MNIST('../../MNIST_data/Train', train=True, download=True)
    testset = MNIST('../../MNIST_data/Test', train=False, download=True)

    # divide data
    train, val, test = divide_cases(trainset, testset)

    # where to save data
    save_path = '../MNIST_data'

    # main loop for each data type
    main(train, save_path, '/train')
    main(val, save_path, '/validate')
    main(test, save_path, '/test')

    print("MNIST loading completed.")