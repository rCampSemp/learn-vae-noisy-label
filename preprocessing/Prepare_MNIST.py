import torch
from torchvision.datasets import MNIST 
import os
import cv2
import numpy as np
from tifffile import imsave

trainset = MNIST('../../MNIST_data/Train', train=True, download=True)
testset = MNIST('../../MNIST_data/Test', train=False, download=True)

def divide_cases(trainset, testset):
    """Divides mnist datasets into train val and test split datasets.

    Args:
        trainset (_type_): MNIST dataset train
        testset (_type_): MNIST dataset test

    Returns:
            - train_set (subset)
            - val_set (subset)
            - testset (Dataset)
    """

    train_set, val_set = torch.utils.data.random_split(trainset, [50000, 10000])
    return train_set, val_set, testset

def split_modes(dataset, split_no=5):
    data_len = len(dataset)
    split_list = [int(data_len/split_no) for _ in range(split_no-1)] 
    # final split to sum to len(dataset)
    split_list_final = split_list + [data_len - sum(split_list)]
    modes = torch.utils.data.random_split(dataset, split_list_final)
    return modes
# guassian 0-1, masks 0 or 1

train_modes, val_modes, test_modes = divide_cases(trainset, testset)


def gauss_transform(img, mean=2, std=5, weight=0.5):
    # img, label
    img = img.copy()
    gaussian = np.uint8(np.random.normal(mean, std, (img.shape[0],img.shape[1])))
    # 0 - 255 to 0 - 1 range
    img = cv2.addWeighted(img, 1.0, gaussian, weight, 0) / 255
    return img

def main(data, save_path, split):
    """main process

    Args:
        data (_type_): _description_
        save_path (_type_): _description_
        split (_type_): _description_
    """

    save_path = save_path + split
    os.makedirs(save_path, exist_ok=True)
    
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

    kernel = np.ones((2, 2), np.uint8)
    
    # labels not needed
    for idx, (img, _) in enumerate(data):
        # opencv compatible type
        img = np.asarray(img, dtype=np.uint8)

        gauss_img = gauss_transform(img)
        # make gauss img 3 channels for compatibility with customdataset_punet
        gauss_img = gauss_img.reshape((28, 28, 1)).repeat(3, -1) # repeat the last (-1) dimension three times

        under_img = cv2.erode(img, kernel, iterations=2)
        over_img = cv2.dilate(img, kernel, iterations=2)

        # morpological operations to produce wrong segmentation
        rows, cols = img.shape
        M = cv2.getRotationMatrix2D(((cols-1)/2.0,(rows-1)/2.0),135,1)
        wrong_img = cv2.warpAffine(img,M,(cols,rows))
        wrong_img = cv2.morphologyEx(wrong_img, cv2.MORPH_TOPHAT, kernel)
        wrong_img = cv2.dilate(wrong_img, kernel, iterations=4)
        wrong_img = cv2.morphologyEx(wrong_img, cv2.MORPH_TOPHAT, kernel)
        wrong_img = cv2.dilate(wrong_img, kernel, iterations=4)

        save_gauss_file = save_gauss + '/' + str(idx) + '_Gaussian.tif'
        save_under_file = save_under + '/' + str(idx) + '_Under.tif'
        save_over_file = save_over + '/' + str(idx) + '_Over.tif'
        save_wrong_file = save_wrong + '/' + str(idx) + '_Wrong.tif'
        save_GT_file = save_GT + '/' + str(idx) + '_GT.tif'


        (_, wrong_img) = cv2.threshold(wrong_img, 50, 1,
            cv2.THRESH_BINARY)
        (_, under_img) = cv2.threshold(under_img, 0, 1,
            cv2.THRESH_BINARY)
        (_, over_img) = cv2.threshold(over_img, 0, 1,
            cv2.THRESH_BINARY)
        (_, GT_img) = cv2.threshold(img, 0, 1,
            cv2.THRESH_BINARY)

        imsave(save_gauss_file, gauss_img)
        imsave(save_GT_file, GT_img)
        imsave(save_over_file, over_img)
        imsave(save_under_file, under_img)
        imsave(save_wrong_file, wrong_img)

if __name__ == '__main__':
    trainset = MNIST('../../MNIST_data/Train', train=True, download=True)
    testset = MNIST('../../MNIST_data/Test', train=False, download=True)

    train, val, test = divide_cases(trainset, testset)
    save_path = '../MNIST_data'
    main(train, save_path, '/train')
    main(val, save_path, '/validate')
    main(test, save_path, '/test')

    print("MNIST loading completed.")