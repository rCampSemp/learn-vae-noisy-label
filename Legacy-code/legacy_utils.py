import os
import glob
import random
import torch
import imageio
import errno
import numpy as np
import tifffile as tiff

import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils import data

from sklearn.metrics import confusion_matrix, jaccard_score


# ================================================================
# Legacy
# ================================================================


def evaluate(data, model, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model.eval()
    # model2.eval()
    #
    test_dice = 0
    # test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model(v_images)
        # b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        # v_outputs_noisy = []
        #
        # v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        # v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        # v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        # for cm in cms:
        #     #
        #     cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
        #     cm = cm / cm.sum(1, keepdim=True)
        #     v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
        #     v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        #     _, v_noisy_output = torch.max(v_noisy_output, dim=1)
        #     v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        # #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        # epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        # v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        # test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1)


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_folder, labels_folder, augmentation):

        # 1. Initialize file paths or a list of file names.
        self.imgs_folder = imgs_folder
        self.labels_folder = labels_folder
        self.data_augmentation = augmentation
        # self.transform = transforms

    def __getitem__(self, index):
        # 1. Read one data from file (e.g. using num py.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).

        all_images = glob.glob(os.path.join(self.imgs_folder, '*.npy'))
        all_labels = glob.glob(os.path.join(self.labels_folder, '*.npy'))
        # sort all in the same order
        all_labels.sort()
        all_images.sort()
        #
        # label = Image.open(all_labels[index])
        # label = tiff.imread(all_labels[index])
        label = np.load(all_labels[index])
        label = np.array(label, dtype='float32')
        # image = tiff.imread(all_images[index])
        image = np.load(all_images[index])
        image = np.array(image, dtype='float32')
        #
        labelname = all_labels[index]
        path_label, labelname = os.path.split(labelname)
        labelname, labelext = os.path.splitext(labelname)
        #
        c_amount = len(np.shape(label))
        #

        #
        # Reshaping everyting to make sure the order: channel x height x width
        if c_amount == 3:
            d1, d2, d3 = np.shape(label)
            if d1 != min(d1, d2, d3):
                label = np.reshape(label, (d3, d1, d2))
                #
        elif c_amount == 2:
            h, w = np.shape(label)
            label = np.reshape(label, (1, h, w))
        #
        d1, d2, d3 = np.shape(image)
        #
        if d1 != min(d1, d2, d3):
            #
            image = np.reshape(image, (d3, d1, d2))
        #
        if self.data_augmentation == 'full':
            # augmentation:
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

        elif self.data_augmentation == 'flip':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation > 0.5 or augmentation == 0.5:
                #
                c, h, w = np.shape(image)
                #
                for channel in range(c):
                    #
                    image[channel, :, :] = np.flip(image[channel, :, :], axis=0).copy()
                    #
                label = np.flip(label, axis=1).copy()

        elif self.data_augmentation == 'all_flip':
            # augmentation:
            augmentation = random.uniform(0, 1)
            #
            if augmentation > 0.5 or augmentation == 0.5:
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

        return image, label, labelname

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(glob.glob(os.path.join(self.imgs_folder, '*.npy')))


# ============================================================================================

def evaluate_noisy_label(data, model1, model2, class_no):

    """

    Args:
        data:
        model1:
        model2:
        class_no:

    Returns:

    """

    model1.eval()
    model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_true, v_imagename) in enumerate(data):
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        v_outputs_logits_noisy = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        for v_noisy_logit in v_outputs_logits_noisy:
            #
            _, v_noisy_output = torch.max(v_noisy_logit, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))

    return test_dice / (i + 1), v_ged

def evaluate_noisy_label_2(data, model1, model2, class_no):

    """

    Args:
        data:
        model1:
        model2:
        class_no:

    Returns:

    """
    model1.eval()
    model2.eval()

    test_dice = 0
    test_dice_all = []

    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_true, v_imagename) in enumerate(data):
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        for cm in cms:
            #
            cm = cm.reshape(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_logit = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_logit, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))

    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_3(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        # v_outputs_logits = v_outputs_logits.permute(0, 2, 3, 1).contiguous()
        # v_outputs_logits = v_outputs_logits.reshape(b * h * w, c, 1)
        #
        for cm in cms:
            #
            cm = cm.reshape(b * h * w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits)
            # v_noisy_output = v_noisy_output.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_4(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_6(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            b, c_r_d, h, w = cm.size()
            r = c_r_d // c // 2
            cm1 = cm[:, 0:r * c, :, :]
            if r == 1:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            else:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
            cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
            #
            v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
            v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            # v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            # v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_7(data, model1, model2, class_no, low_rank):
    """

    Args:
        data:
        model1:
        model2:
        class_no:
        low_rank:

    Returns:

    """
    model1.eval()
    model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            if low_rank is False:
                #
                cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
                cm = cm / cm.sum(1, keepdim=True)
                v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
                v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                #
            else:
                #
                b, c_r_d, h, w = cm.size()
                r = c_r_d // c // 2
                cm1 = cm[:, 0:r * c, :, :]
                cm2 = cm[:, r * c:c_r_d, :, :]
                cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
                cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
                #
                cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
                cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
                #
                v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
                v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
                v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                #
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_5(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_labels_true, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            # cm = cm.reshape(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_true, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged



def evaluate(evaluatedata, model, device, class_no):
    """

    Args:
        evaluatedata:
        model:
        device:
        class_no:

    Returns:

    """
    model.eval()
    #
    with torch.no_grad():
        #
        test_iou = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(evaluatedata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            #
            testoutput = model(testimg)
            if class_no == 2:
                testoutput = torch.sigmoid(testoutput)
                testoutput = (testoutput > 0.5).float()
            else:
                _, testoutput = torch.max(testoutput, dim=1)
            #
            mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)
            test_iou += mean_iu_
        #
        return test_iou / (j+1)

def evaluate_LIDC(evaluatedata, model, device, class_no):
    """

    Args:
        evaluatedata:
        model:
        device:
        class_no:

    Returns:

    """
    model.eval()
    #
    with torch.no_grad():
        #
        test_iou = 0
        #
        for j, (testimg, true_image, annots, imagename) in enumerate(evaluatedata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            #
            testoutput, _ = model(testimg)
            if class_no == 2:
                testoutput = torch.sigmoid(testoutput)
                testoutput = (testoutput > 0.5).float()
            else:
                _, testoutput = torch.max(testoutput, dim=1)
            #
            mean_iu_ = segmentation_scores(true_image.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)
            test_iou += mean_iu_
        #
        return test_iou / (j+1)

def test(testdata,
         model,
         device,
         class_no,
         save_path):
    """

    Args:
        testdata:
        model:
        device:
        class_no:
        save_path:

    Returns:

    """
    model.eval()

    with torch.no_grad():
        #
        test_iou = 0
        #
        for j, (testimg, testlabel, testname) in enumerate(testdata):
            #
            testimg = testimg.to(device=device, dtype=torch.float32)
            testlabel = testlabel.to(device=device, dtype=torch.float32)
            #
            testoutput = model(testimg)
            if class_no == 2:
                testoutput = torch.sigmoid(testoutput)
                testoutput = (testoutput > 0.5).float()
            else:
                _, testoutput = torch.max(testoutput, dim=1)
            #
            mean_iu_ = segmentation_scores(testlabel.cpu().detach().numpy(), testoutput.cpu().detach().numpy(), class_no)
            test_iou += mean_iu_
            #
            # ========================================================
            # # Plotting segmentation:
            # ========================================================
            prediction_map_path = save_path + '/' + 'Visual_results'
            #
            try:
                os.mkdir(prediction_map_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                pass

            b, c, h, w = np.shape(testlabel)

            testoutput_original = np.asarray(testoutput.cpu().detach().numpy(), dtype=np.uint8)
            testoutput_original = np.squeeze(testoutput_original, axis=0)
            testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)
            #
            if class_no == 2:
                segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
                #
                segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
                segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                #
            else:
                segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
                if class_no == 4:
                    # multi class for brats 2018
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                    #
                elif class_no == 8:
                    # multi class for cityscapes
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 255
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 0, testoutput_original[:, :, 1] == 0, testoutput_original[:, :, 2] == 0)] = 0
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 153
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 51
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 4, testoutput_original[:, :, 1] == 4, testoutput_original[:, :, 2] == 4)] = 255
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 255
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 102
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 5, testoutput_original[:, :, 1] == 5, testoutput_original[:, :, 2] == 5)] = 178
                    #
                    segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
                    segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 255
                    segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 6, testoutput_original[:, :, 1] == 6, testoutput_original[:, :, 2] == 6)] = 102
                    #
            prediction_name = 'seg_' + testname[0] + '.png'
            full_error_map_name = os.path.join(prediction_map_path, prediction_name)
            imageio.imsave(full_error_map_name, segmentation_map)

        #
        prediction_result_path = save_path + '/Quantitative_Results'
        #
        try:
            #
            os.mkdir(prediction_result_path)
            #
        except OSError as exc:
            #
            if exc.errno != errno.EEXIST:
                #
                raise
            #
            pass
        #
        result_dictionary = {'Test dice': str(test_iou / len(testdata))}
        #
        ff_path = prediction_result_path + '/test_result_data.txt'
        ff = open(ff_path, 'w')
        ff.write(str(result_dictionary))
        ff.close()

        print('Test iou: {:.4f}, '.format(test_iou / len(testdata)))


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")


def segmentation_scores(label_trues, label_preds, n_class):
    '''
    :param label_trues:
    :param label_preds:
    :param n_class:
    :return:
    '''
    assert len(label_trues) == len(label_preds)

    if n_class == 2:
        #
        output_zeros = np.zeros_like(label_preds)
        output_ones = np.ones_like(label_preds)
        label_preds = np.where((label_preds > 0.5), output_ones, output_zeros)

    label_trues += 1
    label_preds += 1

    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()
    label_preds = label_preds * (label_trues > 0)

    intersection = label_preds * (label_preds == label_trues)
    (area_intersection, _) = np.histogram(intersection, bins=n_class, range=(1, n_class))
    (area_pred, _) = np.histogram(label_preds, bins=n_class, range=(1, n_class))
    (area_lab, _) = np.histogram(label_trues, bins=n_class, range=(1, n_class))
    area_union = area_pred + area_lab
    #
    return ((area_intersection + 1e-6) / (area_union + 1e-6)).mean()


def preprocessing_accuracy(label_true, label_pred, n_class):
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype='int8')
    label_true = np.asarray(label_true, dtype='int8')

    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 8)

    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)

    return label_true, label_pred


def calculate_cm(pred, true):
    #
    pred = pred.view(-1)
    true = true.view(-1)
    #
    pred = pred.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    #
    confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all')
    #
    # if tag == 'brats':
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1, 2, 3])
    # else:
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1])
    #
    #
    return confusion_matrices


# ================================
# Evaluation
# ================================


def evaluate_noisy_label_4(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_5(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_labels_true, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            # cm = cm.reshape(b * h * w, c, c)
            # cm = cm / cm.sum(1, keepdim=True)
            # v_noisy_output = torch.bmm(cm, v_outputs_logits.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
            # cm = cm.permute(0, 2, 3, 1).contiguous().view(b * h * w, c, c)
            cm = cm.reshape(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = seg_score(v_labels_true, v_output.cpu().detach().numpy())
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_true.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged


def evaluate_noisy_label_6(data, model1, class_no):
    """

    Args:
        data:
        model1:
        class_no:

    Returns:

    """
    model1.eval()
    # model2.eval()
    #
    test_dice = 0
    test_dice_all = []
    #
    for i, (v_images, v_labels_over, v_labels_under, v_labels_wrong, v_labels_good, v_imagename) in enumerate(data):
        #
        # print(i)
        #
        v_images = v_images.to(device='cuda', dtype=torch.float32)
        v_outputs_logits, cms = model1(v_images)
        b, c, h, w = v_outputs_logits.size()
        v_outputs_logits = nn.Softmax(dim=1)(v_outputs_logits)
        # cms = model2(v_images)
        #
        _, v_output = torch.max(v_outputs_logits, dim=1)
        v_outputs_noisy = []
        #
        v_outputs_logits = v_outputs_logits.view(b, c, h*w)
        v_outputs_logits = v_outputs_logits.permute(0, 2, 1).contiguous().view(b*h*w, c)
        v_outputs_logits = v_outputs_logits.view(b * h * w, c, 1)
        #
        for cm in cms:
            #
            b, c_r_d, h, w = cm.size()
            r = c_r_d // c // 2
            cm1 = cm[:, 0:r * c, :, :]
            if r == 1:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            else:
                cm2 = cm[:, r * c:c_r_d-1, :, :]
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
            cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
            #
            v_noisy_output = torch.bmm(cm1_reshape, v_outputs_logits)
            v_noisy_output = torch.bmm(cm2_reshape, v_noisy_output).view(b * h * w, c)
            v_noisy_output = v_noisy_output.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            # v_noisy_output = torch.bmm(cm, v_outputs_logits).view(b*h*w, c)
            # v_noisy_output = v_noisy_output.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output, dim=1)
            v_outputs_noisy.append(v_noisy_output.cpu().detach().numpy())
        #
        v_dice_ = segmentation_scores(v_labels_good, v_output.cpu().detach().numpy(), class_no)
        #
        epoch_noisy_labels = [v_labels_over.cpu().detach().numpy(), v_labels_under.cpu().detach().numpy(), v_labels_wrong.cpu().detach().numpy(), v_labels_good.cpu().detach().numpy()]
        v_ged = generalized_energy_distance(epoch_noisy_labels, v_outputs_noisy, class_no)
        test_dice += v_dice_
        test_dice_all.append(test_dice)
        #
    # print(i)
    # print(test_dice)
    # print(test_dice / (i + 1))
    #
    return test_dice / (i + 1), v_ged
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
