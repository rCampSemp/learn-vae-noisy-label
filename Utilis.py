import os
import glob
import random
import torch
import errno
import numpy as np
import tifffile as tiff

import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import jaccard_score

# =============================================


class CustomDataset_punet(torch.utils.data.Dataset):
    """Dataset class for probabilistic U-Net, used in our work for MNIST.

    Args:
        dataset_location (str): file path to location of dataset
        dataset_tag (str): dataset in use. 'mnist' for mnist use.
        noisylabel (str): type of noisy label regime to use.
        augmentation (bool, optional): if True, performs random flipping of image as augmentation. Defaults to False.
    """
    def __init__(self, dataset_location, dataset_tag, noisylabel, augmentation=False):
        """Constructor
        """
        #
        self.label_mode = noisylabel
        self.dataset_tag = dataset_tag
        ## define folder paths
        if noisylabel == 'multi':
            #
            if dataset_tag == 'mnist':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/GT'
                self.image_folder = dataset_location + '/Gaussian'
            elif dataset_tag == 'brats':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/Good'
                self.image_folder = dataset_location + '/Image'
            elif dataset_tag == 'lidc':
                self.label_over_folder = dataset_location + '/Annotator_1'
                self.label_under_folder = dataset_location + '/Annotator_2'
                self.label_wrong_folder = dataset_location + '/Annotator_3'
                self.label_good_folder = dataset_location + '/Annotator_4'
                self.label_true_folder = dataset_location + '/Annotator_5'
                self.image_folder = dataset_location + '/Image'
                #
        elif noisylabel == 'binary':
            if dataset_tag == 'mnist':
                self.label_folder = dataset_location + '/Mean'
                self.image_folder = dataset_location + '/Gaussian'
                self.true_label_folder = dataset_location + '/GT'

        elif noisylabel == 'normal':
            if dataset_tag == 'mnist':
                self.label_folder = dataset_location + '/GT'
                self.image_folder = dataset_location + '/Gaussian'

        elif noisylabel == 'p_unet':
            if dataset_tag == 'mnist':
                self.label_over_folder = dataset_location + '/Over'
                self.label_under_folder = dataset_location + '/Under'
                self.label_wrong_folder = dataset_location + '/Wrong'
                self.label_good_folder = dataset_location + '/GT'

                self.label_folder = dataset_location + '/All'
                self.image_folder = dataset_location + '/Gaussian'

        self.data_aug = augmentation

    def __getitem__(self, index):

        if self.label_mode == 'multi':
            #
            if self.dataset_tag == 'mnist' or self.dataset_tag == 'brats':
                # get file paths for all data
                all_labels_over = glob.glob(os.path.join(self.label_over_folder, '*.tif'))
                all_labels_over.sort()
                #
                all_labels_under = glob.glob(os.path.join(self.label_under_folder, '*.tif'))
                all_labels_under.sort()
                #
                all_labels_wrong = glob.glob(os.path.join(self.label_wrong_folder, '*.tif'))
                all_labels_wrong.sort()
                #
                all_labels_good = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                all_labels_good.sort()
                #
                all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
                all_images.sort()
                #
                label_over = tiff.imread(all_labels_over[index])
                label_over = np.array(label_over, dtype='float32')
                #
                label_under = tiff.imread(all_labels_under[index])
                label_under = np.array(label_under, dtype='float32')
                #
                label_wrong = tiff.imread(all_labels_wrong[index])
                label_wrong = np.array(label_wrong, dtype='float32')
                #
                label_good = tiff.imread(all_labels_good[index])
                label_good = np.array(label_good, dtype='float32')
                #
                image = tiff.imread(all_images[index])
                image = np.array(image, dtype='float32')
                #

                label_over[label_over == 4.0] = 3.0
                label_wrong[label_wrong == 4.0] = 3.0
                label_good[label_good == 4.0] = 3.0
                label_under[label_under == 4.0] = 3.0

                if self.dataset_tag == 'mnist':
                    label_over = np.where(label_over > 0.5, 1.0, 0.0)
                    label_under = np.where(label_under > 0.5, 1.0, 0.0)
                    label_wrong = np.where(label_wrong > 0.5, 1.0, 0.0)

                    if np.amax(label_good) != 1.0:
                        # sometimes, some preprocessing might give it as 0 - 255 range
                        label_good = np.where(label_good > 10.0, 1.0, 0.0)
                    else:
                        assert np.amax(label_good) == 1.0
                        label_good = np.where(label_good > 0.5, 1.0, 0.0)

                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_over))
                # Reshaping everyting to make sure the order: channel x height x width
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_over)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_over = np.transpose(label_over, (2, 0, 1))
                        label_under = np.transpose(label_under, (2, 0, 1))
                        label_wrong = np.transpose(label_wrong, (2, 0, 1))
                        label_good = np.transpose(label_good, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_over = np.expand_dims(label_over, axis=0)
                    label_under = np.expand_dims(label_under, axis=0)
                    label_wrong = np.expand_dims(label_wrong, axis=0)
                    label_good = np.expand_dims(label_good, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_over = np.flip(label_over, axis=1).copy()
                        label_over = np.flip(label_over, axis=2).copy()
                        label_under = np.flip(label_under, axis=1).copy()
                        label_under = np.flip(label_under, axis=2).copy()
                        label_wrong = np.flip(label_wrong, axis=1).copy()
                        label_wrong = np.flip(label_wrong, axis=2).copy()
                        label_good = np.flip(label_good, axis=1).copy()
                        label_good = np.flip(label_good, axis=2).copy()
                        #
                return image, label_over, label_under, label_wrong, label_good, imagename

            elif self.dataset_tag == 'lidc':
                #
                all_labels_over = glob.glob(os.path.join(self.label_over_folder, '*.tif'))
                all_labels_over.sort()
                #
                all_labels_under = glob.glob(os.path.join(self.label_under_folder, '*.tif'))
                all_labels_under.sort()
                #
                all_labels_wrong = glob.glob(os.path.join(self.label_wrong_folder, '*.tif'))
                all_labels_wrong.sort()
                #
                all_labels_good = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                all_labels_good.sort()
                #
                all_labels_true = glob.glob(os.path.join(self.label_true_folder, '*.tif'))
                all_labels_true.sort()
                #
                all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
                all_images.sort()
                #
                label_over = tiff.imread(all_labels_over[index])
                label_over = np.array(label_over, dtype='float32')
                #
                label_under = tiff.imread(all_labels_under[index])
                label_under = np.array(label_under, dtype='float32')
                #
                label_wrong = tiff.imread(all_labels_wrong[index])
                label_wrong = np.array(label_wrong, dtype='float32')
                #
                label_good = tiff.imread(all_labels_good[index])
                label_good = np.array(label_good, dtype='float32')
                #
                label_true = tiff.imread(all_labels_true[index])
                label_true = np.array(label_true, dtype='float32')
                #
                image = tiff.imread(all_images[index])
                image = np.array(image, dtype='float32')
                #
                # dim_length = len(np.shape(label_over))

                # label_over[label_over == 4.0] = 3.0
                # label_wrong[label_wrong == 4.0] = 3.0
                # label_good[label_good == 4.0] = 3.0
                # label_under[label_under == 4.0] = 3.0
                # label_true[label_true == 4.0] = 3.0
                # print(np.unique(label_over))
                # label_over: h x w
                # image: h x w x c
                c_amount = len(np.shape(label_over))
                # Reshaping everyting to make sure the order: channel x height x width
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(label_over)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        assert d3 == min(d1, d2, d3)
                        #
                        label_over = np.transpose(label_over, (2, 0, 1))
                        label_under = np.transpose(label_under, (2, 0, 1))
                        label_wrong = np.transpose(label_wrong, (2, 0, 1))
                        label_good = np.transpose(label_good, (2, 0, 1))
                        label_true = np.transpose(label_true, (2, 0, 1))
                    #
                elif c_amount == 2:
                    #
                    label_over = np.expand_dims(label_over, axis=0)
                    label_under = np.expand_dims(label_under, axis=0)
                    label_wrong = np.expand_dims(label_wrong, axis=0)
                    label_good = np.expand_dims(label_good, axis=0)
                    label_true = np.expand_dims(label_true, axis=0)
                #
                c_amount = len(np.shape(image))
                #
                if c_amount == 3:
                    #
                    d1, d2, d3 = np.shape(image)
                    #
                    if d1 != min(d1, d2, d3):
                        #
                        image = np.transpose(image, (2, 0, 1))
                        #
                elif c_amount == 2:
                    #
                    image = np.expand_dims(image, axis=0)
                #
                imagename = all_images[index]
                path_image, imagename = os.path.split(imagename)
                imagename, imageext = os.path.splitext(imagename)
                #
                if self.data_aug is True:
                    #
                    augmentation = random.uniform(0, 1)
                    #
                    if augmentation > 0.5:
                        #
                        c, h, w = np.shape(image)
                        #
                        for channel in range(c):
                            #
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                            image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                            #
                        label_over = np.flip(label_over, axis=1).copy()
                        label_over = np.flip(label_over, axis=2).copy()
                        label_under = np.flip(label_under, axis=1).copy()
                        label_under = np.flip(label_under, axis=2).copy()
                        label_wrong = np.flip(label_wrong, axis=1).copy()
                        label_wrong = np.flip(label_wrong, axis=2).copy()
                        label_good = np.flip(label_good, axis=1).copy()
                        label_good = np.flip(label_good, axis=2).copy()
                        label_true = np.flip(label_true, axis=1).copy()
                        label_true = np.flip(label_true, axis=2).copy()
                        #
                return image, label_over, label_under, label_wrong, label_good, label_true, imagename
        #
        elif self.label_mode == 'binary':

            all_true_labels = glob.glob(os.path.join(self.true_label_folder, '*.tif'))
            all_true_labels.sort()
            all_labels = glob.glob(os.path.join(self.label_folder, '*.tif'))
            all_labels.sort()
            all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
            all_images.sort()
            #
            image = tiff.imread(all_images[index])
            image = np.array(image, dtype='float32')
            #
            label = tiff.imread(all_labels[index])
            label = np.array(label, dtype='float32')
            #
            true_label = tiff.imread(all_true_labels[index])
            true_label = np.array(true_label, dtype='float32')
            #
            d1, d2, d3 = np.shape(label)
            image = np.reshape(image, (d3, d1, d2))
            label = np.reshape(label, (d3, d1, d2))
            true_label = np.reshape(true_label, (d3, d1, d2))
            #
            imagename = all_images[index]
            path_image, imagename = os.path.split(imagename)
            imagename, imageext = os.path.splitext(imagename)
            #
            if self.data_aug is True:
                #
                augmentation = random.uniform(0, 1)
                #
                if augmentation < 0.25:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                        #
                    label = np.flip(label, axis=1).copy()
                    label = np.flip(label, axis=2).copy()
                    #
                    true_label = np.flip(true_label, axis=1).copy()
                    true_label = np.flip(true_label, axis=2).copy()
                    #
                elif augmentation < 0.5:
                    #
                    mean = 0.0
                    sigma = 0.15
                    noise = np.random.normal(mean, sigma, image.shape)
                    mask_overflow_upper = image + noise >= 1.0
                    mask_overflow_lower = image + noise < 0.0
                    noise[mask_overflow_upper] = 1.0
                    noise[mask_overflow_lower] = 0.0
                    image += noise

                elif augmentation < 0.75:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        channel_ratio = random.uniform(0, 1)
                        #
                        image[channel, :, :] = image[channel, :, :] * channel_ratio

            return image, label, true_label, imagename

        elif self.label_mode == 'p_unet':
            if self.dataset_tag == 'mnist':
                labeltype = np.random.randint(3)
                if labeltype == 0:
                    all_labels = glob.glob(os.path.join(self.label_over_folder, '*.tif'))
                    all_labels.sort()
                elif labeltype == 1:
                    all_labels = glob.glob(os.path.join(self.label_under_folder, '*.tif'))
                    all_labels.sort()
                elif labeltype == 2:
                    all_labels = glob.glob(os.path.join(self.label_wrong_folder, '*.tif'))
                    all_labels.sort()
                elif labeltype == 3:
                    all_labels = glob.glob(os.path.join(self.label_good_folder, '*.tif'))
                    all_labels.sort()
            #
            # all_labels = glob.glob(os.path.join(self.label_folder, '*.tif'))
            # all_labels.sort()
            all_images = glob.glob(os.path.join(self.image_folder, '*.tif'))
            all_images.sort()
            #
            image = tiff.imread(all_images[index])
            image = np.array(image, dtype='float32')
            #
            label = tiff.imread(all_labels[index])
            label = np.array(label, dtype='float32')
            #
            d1, d2, d3 = np.shape(image)
            image = np.reshape(image, (d3, d1, d2))
            label = np.reshape(label, (1, d1, d2))
            #
            imagename = all_images[index]
            path_image, imagename = os.path.split(imagename)
            imagename, imageext = os.path.splitext(imagename)
            #
            if self.data_aug is True:
                #
                augmentation = random.uniform(0, 1)
                #
                if augmentation > 0.5:
                    #
                    c, h, w = np.shape(image)
                    #
                    for channel in range(c):
                        #
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                        image[channel, :, :] = np.flip(image[channel, :, :], axis=1).copy()
                        #
                    label = np.flip(label, axis=1).copy()
                    label = np.flip(label, axis=2).copy()
                    #
                # elif augmentation < 0.5:
                #     #
                #     mean = 0.0
                #     sigma = 0.15
                #     noise = np.random.normal(mean, sigma, image.shape)
                #     mask_overflow_upper = image + noise >= 1.0
                #     mask_overflow_lower = image + noise < 0.0
                #     noise[mask_overflow_upper] = 1.0
                #     noise[mask_overflow_lower] = 0.0
                #     image += noise
                #
                # elif augmentation < 0.75:
                #     #
                #     c, h, w = np.shape(image)
                #     #
                #     for channel in range(c):
                #         #
                #         channel_ratio = random.uniform(0, 1)
                #         #
                #         image[channel, :, :] = image[channel, :, :] * channel_ratio

            return image, label, imagename

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.image_folder, '*.tif')))


class CustomDataset_LIDC(torch.utils.data.Dataset):
    """Dataset class for LIDC data

    Args:
        dataset_location (str): file path to our processed lidc data
        model (str, optional): either punet or ours, denotes getitem return scheme model depending on model. Defaults to 'ours'
        augmentation (bool, optional): True for random flipping of the output images or False for no augmentation at all. Defaults to False
    """
    def __init__(self, dataset_location, model='ours', augmentation=False):
        """Constructor
        """
        self.annot_path = dataset_location + '/masks/annots'
        self.truth_path = dataset_location + '/masks/GT'
        self.scan_path = dataset_location + '/scans'
        self.data_aug = augmentation
        self.modeltype = model

    def __getitem__(self, index):

        ## get paths for all data
        all_true_images = glob.glob(os.path.join(self.truth_path, '*.tif'))
        all_true_images.sort()
        #
        all_images = glob.glob(os.path.join(self.scan_path, '*.tif'))
        all_images.sort()
        #
        all_annot_images = glob.glob(os.path.join(self.annot_path, '*.tif'))
        all_annot_images.sort()
        #
        annot_image = tiff.imread(all_annot_images[index])
        annot_image = np.array(annot_image, dtype='float32')
        #
        true_image = tiff.imread(all_true_images[index])
        true_image = np.array(true_image, dtype='float32')
        #
        image = tiff.imread(all_images[index])
        image = np.array(image, dtype='float32')
        #
        ## data augmentation randomly flip images
        if self.data_aug is True:
            #
            aug_thresh = random.uniform(0, 1) # get random float
            #
            if aug_thresh > 0.5:   
                #
                true_image = np.flip(true_image, axis=1).copy()
                true_image = np.flip(true_image, axis=2).copy()
                #
                image = np.flip(image, axis=1).copy()
                image = np.flip(image, axis=2).copy()
                #
                annot_image = np.flip(annot_image, axis=1).copy()
                annot_image = np.flip(annot_image, axis=2).copy()
        
        # name of scan and nodule for returning image name
        imagename = all_images[index]
        path_image, imagename = os.path.split(imagename)
        imagename, imageext = os.path.splitext(imagename)

        # for our proposed model want ground truth and all annotators
        if self.modeltype=='ours':
            return image, true_image, annot_image, imagename

        # no ground truth and one random label for probabilistc unet baseline
        elif self.modeltype=='punet':
            annot = annot_image[:,:,:,np.random.choice(annot_image.shape[3])]
            return image, annot, imagename
        
    
    def __len__(self):
        return len(glob.glob(os.path.join(self.scan_path, '*.tif')))


def LIDC_collate(batch):
    """
    Function that takes in a batch of data from LIDC custom dataset 

    repeats a random tensor along axis in batch of annotations to create 
    stacked batch of equal dimensions

    Args:
        batch (list of tuples of ndarrays): zipped input from batchsize number of LIDC custom dataset data points

    Returns:
        b_images (torch.DoubleTensor): stacked batch of slices of patient scans
        b_true_image (torch.DoubleTensor): stacked batch of slices of ground truth segmentation masks of nodule
        b_annots (torch.DoubleTensor): stacked batch of slices of annotation segmentation masks associated with nodule
        b_imagename (tuple): tuple of str associated with batched slices patient of scan annotation 
    """
    b_images, b_true_image, b_annots, b_imagename = zip(*batch)

    max_sz = max([annots.shape[3] for annots in b_annots])

    b_lst_annots = []

    # repeat annot in each stack of annots so all annots in batch same dim 
    for annots in b_annots: 
        annots = torch.DoubleTensor(annots)

        # get number of annots to fill
        annot_sz_to_fill = max_sz - annots.size(dim=3)

        # if no annots to fill ignore
        if annot_sz_to_fill == 0:
            b_lst_annots.append(annots)
            continue

        annot_to_add = []
        # get random choice of annotation to add to index with all annotations to add
        ann_samples = np.random.choice(annots.size(dim=3), annot_sz_to_fill)
        for i in ann_samples:
            annot_to_add.append(annots[:,:,:,i])

        # stack annotations to add then concatenate to correct axis
        stck_annots_to_add = torch.stack(annot_to_add, 3)
        stck_annots = torch.cat((annots, stck_annots_to_add), 3)
        b_lst_annots.append(stck_annots) 

    # collate input images and ground truth
    b_images = [torch.DoubleTensor(images) for images in b_images]
    b_true_image = [torch.DoubleTensor(true_image) for true_image in b_true_image]

    # if more than one annotation
    if len(b_lst_annots) > 1:
        b_annots = torch.stack(b_lst_annots, 0)
        b_images = torch.stack(b_images, 0)
        b_true_image = torch.stack(b_true_image, 0)

    else:
        b_annots = b_lst_annots[0].unsqueeze(0)
        b_images = b_images[0].unsqueeze(0)
        b_true_image = b_true_image[0].unsqueeze(0)


    return b_images, b_true_image, b_annots, b_imagename


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        # truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        # truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg




def test_punet(net, testdata, save_path, sampling_times, datatag):
    """Testing regime for probabilistic unet model

    Args:
        net (nn.Module): prob unet model
        testdata (torch.utils.data.DataLoader): dataloader for testing data
        save_path (str): file path for saving results
        sampling_times (int): how many samples to generate from model
        datatag (str): 'lidc' for lidc data or 'mnist' for mnist data
    """
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.eval()
    test_dice = []
    test_generalized_energy_distance = []
    epoch_noisy_labels = []
    epoch_noisy_segs = []
    #
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    save_path = save_path + '/Visual_segmentation'
    #
    try:
        os.mkdir(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    with torch.no_grad():
        if datatag == 'mnist':
            for no_eval, (patch_eval, mask_eval_over, mask_eval_under, mask_eval_wrong, mask_eval_good, mask_name_eval) in enumerate(testdata):
                #
                if no_eval < 30:
                    #
                    patch_eval = patch_eval.to(device)
                    mask_eval_over = mask_eval_over.to(device)
                    mask_eval_under = mask_eval_under.to(device)
                    mask_eval_wrong = mask_eval_wrong.to(device)
                    mask_eval_good = mask_eval_good.to(device)
                    #
                    for j in range(sampling_times):
                        #
                        net.eval()
                        # segm input doesn't matter
                        net.forward(patch_eval, mask_eval_good, training=False)
                        seg_sample = net.sample(testing=True)
                        seg_sample = (torch.sigmoid(seg_sample) > 0.5).float()
                        (b, c, h, w) = seg_sample.shape
                        #
                        if j == 0:
                            seg_evaluate = seg_sample
                        else:
                            seg_evaluate += seg_sample
                            #
                        epoch_noisy_segs.append(seg_sample.reshape(28,28).cpu().detach().numpy().astype(bool)) # for ged calculation and saving img
                        # save reconstructed CMs for first 10 images 
                        if no_eval < 10:
                            #
                            save_name = save_path + '/test_' + str(no_eval) + '_sample_' + str(j) + '_seg.png'
                            #
                            plt.imsave(save_name, seg_sample.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                    #
                    seg_evaluate = seg_evaluate / sampling_times
                # save first 10 ground truths
                if no_eval < 10:
                    #
                    gt_save_name = save_path + '/gt_' + str(no_eval) + '.png'
                    #
                    plt.imsave(gt_save_name, mask_eval_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')

                # calculate ged with all annotator noisy labels
                epoch_noisy_labels = [mask_eval_good.reshape(28,28).cpu().detach().numpy().astype(bool), mask_eval_over.reshape(28,28).cpu().detach().numpy().astype(bool), mask_eval_under.reshape(28,28).cpu().detach().numpy().astype(bool), mask_eval_wrong.reshape(28,28).cpu().detach().numpy().astype(bool)]
                ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs)
                
                # jaccard score and convert into dice score
                jac = jaccard_score(mask_eval_good.reshape(28,28).cpu().detach().numpy().astype(bool), seg_evaluate.reshape(28, 28).cpu().detach().numpy().astype(bool), average="micro") 
                dice = 2 * jac / (jac + 1)  

                test_dice.append(dice)
                test_generalized_energy_distance.append(ged)
                

        if datatag == 'lidc':
            for no_eval, (patch_eval, mask_eval_true, mask_eval_annots, mask_name_eval) in enumerate(testdata):
                if no_eval < 30:
                    patch_eval = patch_eval.to(device, dtype=torch.float32)
                    mask_eval_annots = mask_eval_annots.to(device, dtype=torch.float32)
                    mask_eval_true = mask_eval_true.to(device, dtype=torch.float32)
                    
                    sampling_no = mask_eval_annots.shape[4]
                    epoch_noisy_segs = []
                    for j in range(sampling_no):
                        net.eval()
                        # segm input doesn't matter
                        net.forward(patch_eval, mask_eval_annots[:,:,:,:,0], training=False)
                        seg_sample = net.sample(testing=True)
                        seg_sample = (torch.sigmoid(seg_sample) > 0.5).float()
                        (b, c, h, w) = seg_sample.shape
                        #
                        if j == 0:
                            #
                            seg_evaluate = seg_sample
                            #
                        else:
                            #
                            seg_evaluate += seg_sample
                            #
                        epoch_noisy_segs.append(seg_sample.reshape(h,w).cpu().detach().numpy().astype(bool))
                        #
                        if no_eval < 10:
                            #
                            save_name = save_path + '/test_' + str(no_eval) + '_sample_' + str(j) + '_seg.png'
                            #
                            plt.imsave(save_name, seg_sample.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                        #
                    if no_eval < 10:
                        #
                        gt_save_name = save_path + '/gt_' + str(no_eval) + '.png'
                        #
                        plt.imsave(gt_save_name, mask_eval_true.reshape(h, w).cpu().detach().numpy(), cmap='gray')

                    seg_evaluate = seg_evaluate / sampling_no
                    #
                    # val_iou = seg_score(mask_eval_true.reshape(h, w).cpu().detach().numpy(), seg_evaluate.reshape(h,w).cpu().detach().numpy())
                    jac = jaccard_score(mask_eval_true.reshape(h,w).cpu().detach().numpy().astype(bool), seg_evaluate.reshape(h, w).cpu().detach().numpy().astype(bool), average="micro") 
                    dice = 2 * jac / (jac + 1)  
                    test_dice.append(dice)

                    if sampling_no > 1:
                        epoch_noisy_labels = [mask_eval_annots[:,:,:,:,annot_idx].reshape(h, w).numpy() for annot_idx in range(sampling_no)]
                        ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs)
                        test_generalized_energy_distance.append(ged)

    test_dice = eval_metric(test_dice)
    test_generalized_energy_distance = eval_metric(test_generalized_energy_distance)
    #
    result_dictionary = {'Test IoU': str(test_dice), 'Test GED': str(test_generalized_energy_distance)}
    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    print('Test iou: ' + str(test_dice))
    print('Test generalised energy distance: ' + str(test_generalized_energy_distance))


def evaluate_punet(net, val_data, class_no, sampling_no, datatag='mnist'):
    """validating regime for probabilistic unet model

    Args:
        net (nn.Module): prob unet model
        val_data (torch.utils.data.DataLoader): dataloader for validation data
        save_path (str): file path for saving results
        class_no (int): number of classes, 2 for binary segmentaiton.
        sampling_no (int): how many samples to generate from model
        datatag (str, optional): 'lidc' for lidc data or 'mnist' for mnist data. Defaults to 'mnist'

    Returns:
        val_dice (2-tuple of float): 1- mean dice score 2- std dev 
        val_ged (2-tuple of float): 1- mean generalized energy distance 2- std dev 
    """
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate_dice = []
    generalized_energy_distance_epoch = []
    #
    if datatag == 'mnist':
        for no_eval, (patch_eval, mask_eval_over, mask_eval_under, mask_eval_wrong, mask_eval_true, mask_name_eval) in enumerate(val_data):
            #
            patch_eval = patch_eval.to(device)
            mask_eval_over = mask_eval_over.to(device)
            mask_eval_under = mask_eval_under.to(device)
            mask_eval_wrong = mask_eval_wrong.to(device)
            mask_eval_true = mask_eval_true.to(device)
            epoch_noisy_segs = []
            #
            for j in range(sampling_no):
                net.eval()
                # segm input doesn't matter
                net.forward(patch_eval, mask_eval_wrong, training=False)
                seg_sample = net.sample(testing=True)
                seg_sample = (torch.sigmoid(seg_sample) > 0.5).float()
                #
                if j == 0:
                    #
                    seg_evaluate = seg_sample
                    #
                else:
                    #
                    seg_evaluate += seg_sample
                    #
                epoch_noisy_segs.append(seg_sample.reshape(28,28).cpu().detach().numpy().astype(bool)) # for ged calculation
                #
            seg_evaluate = seg_evaluate / sampling_no
            #
            # calculate ged using all noisy annotator labels
            epoch_noisy_labels = [mask_eval_over.reshape(28,28).cpu().detach().numpy().astype(bool), mask_eval_under.reshape(28,28).cpu().detach().numpy().astype(bool), mask_eval_wrong.reshape(28,28).cpu().detach().numpy().astype(bool)]
            ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs)
            
            # jaccard score and convert into dice score
            jac = jaccard_score(mask_eval_true.reshape(28,28).cpu().detach().numpy().astype(bool), seg_evaluate.reshape(28, 28).cpu().detach().numpy().astype(bool), average="micro") 
            val_dice = 2 * jac / (jac + 1)

            validate_dice.append(val_dice)
            generalized_energy_distance_epoch.append(ged)
    
    if datatag == 'lidc':
        for no_eval, (patch_eval, mask_eval_true, mask_eval_annots, mask_name_eval) in enumerate(val_data):
            patch_eval = patch_eval.to(device, dtype=torch.float32)
            mask_eval_annots = mask_eval_annots.to(device, dtype=torch.float32)
            mask_eval_true = mask_eval_true.to(device, dtype=torch.float32)
            
            sampling_no = mask_eval_annots.shape[4]

            # iterate for each sample from network 
            epoch_noisy_segs = []
            for j in range(sampling_no):
                net.eval()
                # segm input doesn't matter
                net.forward(patch_eval, mask_eval_annots[:,:,:,:,0], training=False)
                seg_sample = net.sample(testing=True)
                seg_sample = (torch.sigmoid(seg_sample) > 0.5).float() # threshold iage
                # 
                if j == 0:
                    #
                    seg_evaluate = seg_sample
                    #
                else:
                    #
                    seg_evaluate += seg_sample
                    #
                epoch_noisy_segs.append(seg_sample.reshape(64,64).cpu().detach().numpy().astype(bool)) # for ged calculation
                #
            seg_evaluate = seg_evaluate / sampling_no
            #
            # jaccard score and convert into dice score
            jac = jaccard_score(mask_eval_true.reshape(64,64).cpu().detach().numpy().astype(bool), seg_evaluate.reshape(64, 64).cpu().detach().numpy().astype(bool), average="micro") 
            val_dice = 2 * jac / (jac + 1)
            validate_dice.append(val_dice)

            # get list of annotators for generalized energy distance calc
            if sampling_no > 1:
                epoch_noisy_labels = [mask_eval_annots[:,:,:,:,annot_idx].reshape(64, 64).numpy() for annot_idx in range(sampling_no)]
                ged = generalized_energy_distance(epoch_noisy_labels, epoch_noisy_segs)
                generalized_energy_distance_epoch.append(ged)

    # main and std of metrics
    val_dice = eval_metric(validate_dice)
    val_ged = eval_metric(generalized_energy_distance_epoch)
    #
    return val_dice, val_ged



def seg_score(target, inputs):
    """Calculates dice similarity coefficnet between target and input. 
    Not a global dice score for a more representative score

    Args:
        target (numpy array): ground truth img 
        inputs (numpy array): our reconstructed img

    Returns:
        dice (float): average dice score of batch
    """
    num = target.shape[0]

    # copy arrays
    inputs = np.asarray(inputs, dtype='uint8').copy()
    target = np.asarray(target, dtype='uint8').copy()

    # reshape 
    inputs = inputs.reshape(num, -1)
    target = target.reshape(num, -1)

    # calc intersection and union for dice score
    intersection = (inputs * target).sum(1)
    union = inputs.sum(1) + target.sum(1)

    # dice formula with small constnat to avoid zero error
    dice = (2. * intersection) / (union + 1e-8)

    # calc average
    dice = dice.sum()/num

    return dice

def generalized_energy_distance(all_gts, all_segs):
    """Function to calculate generalized energy distance

    Args:
        all_gts (list): list of all noisy labels as torch.Tensor 
        all_segs (list): list of reconstructed ground truths from our model as torch.Tensor

    Returns:
        ged_metric (float): calculated generalized energy distance metric
    """
    # This is slightly different from the original paper:
    # We didn't take the distance to the power of 2
    #
    # (1 - iou) metric for all combinations of input noisy labels
    gt_gt_dist = [1-jaccard_score(gt_1, gt_2, average="micro") for i1, gt_1 in enumerate(all_gts) for i2, gt_2 in enumerate(all_gts) if i1 != i2]

    # (1 - iou) metric for all combinations of reconstructed segmentations
    seg_seg_dist = [1-jaccard_score(seg_1, seg_2, average="micro") for i1, seg_1 in enumerate(all_segs) for i2, seg_2 in enumerate(all_segs) if i1 != i2]

    # (1 - iou) metric for all combinations of reconstructed segmentations and noisy labels
    seg_gt_list = [1-jaccard_score(seg_, gt_, average="micro") for i, seg_ in enumerate(all_segs) for j, gt_ in enumerate(all_gts)]

    # 
    ged_metric = sum(gt_gt_dist) / len(gt_gt_dist) + sum(seg_seg_dist) / len(seg_seg_dist) + 2 * sum(seg_gt_list) / len(seg_gt_list)
    return ged_metric


def eval_metric(all_values):
    """Function to evaluate a list of metrics.

    Args:
        all_values (list): list of metrics as float or int 

    Returns:
        mean (float): mean of all_values list
        std_err (float): standard deviation of all_values list
    """
    mean = np.mean(all_values)
    std = np.std(all_values)

    std_err = std / np.sqrt(len(all_values))
    return mean, std_err

