import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def deterministic_noisy_label_loss(pred, cm, labels, epoch, total_epoch, data='lidc', ramp_up=0.5):
    """
    Under construction
    """
    # regularisation = 0.0
    b, c, h, w = pred.size()

    # b*c x h*w ---> b*h*w x c x 1
    pred_noisy = pred.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
    # label_noisy: noisy label, b x h x w
    # convex combination of noisy labels:
    # weights = np.random.dirichlet((1, 1, 1, 1), size=1)
    # for i, each_label in enumerate(labels):
    #     if i == 0:
    #         label = each_label*weights[0][i]
    #     else:
    #         label += each_label*weights[0][i]
    # label = (label > 0.5).float()
    # random choice:
    if data == 'lidc':
        label = labels[:,:,:,:,np.random.choice(labels.shape[4])]
    elif data == 'mnist':
        label = random.choice(labels)

    # loss = 0
    # kld_loss = 0
    # b x c**2 x h x w ---> b*h*w x c x c
    anti_corrpution_cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # normalisation along the rows:
    anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)

    # matrix multiplication to calculate the predicted noisy segmentation:
    # cm: b*h*w x c x c
    # pred_clean: b*h*w x c x 1
    pred_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b*h*w, c)
    pred_clean = pred_clean.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    _, pseudo_labels = torch.max(pred, dim=1)
    ramp_up_threshold = int(total_epoch*ramp_up)

    if epoch < ramp_up_threshold:
        beta_current = epoch / ramp_up_threshold
        loss = beta_current*nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) + (1 - beta_current)*nn.CrossEntropyLoss(reduction='mean')(pred_clean, label.view(b, h, w).long()) + nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())
    else:
        loss = nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) + nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())
        
    return loss

def dice_loss(input, target):
    """ This is a normal dice loss function for binary segmentation.

    Args:
        input: output of the segmentation network
        target: ground truth label

    Returns:
        dice score

    """
    smooth = 1

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


