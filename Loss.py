import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def stochastic_noisy_label_loss(pred, cm, mu, logvar, labels, epoch, total_epoch, ramp_up=0.5, alpha=1.0):
    """ This function defines the proposed trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.

    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation

    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss

    """
    # regularisation = 0.0
    b, c, h, w = pred.size()

    # normalise the segmentation output tensor along dimension 1
    pred_norm_prob = nn.Softmax(dim=1)(pred)

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm_prob.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

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
    label = random.choice(labels)

    # b x c**2 x h x w ---> b*h*w x c x c
    cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # normalisation along the rows:
    # cm = cm / cm.sum(1, keepdim=True)
    cm = torch.softmax(cm, dim=1)

    # matrix multiplication to calculate the predicted noisy segmentation:
    # cm: b*h*w x c x c
    # pred_noisy: b*h*w x c x 1
    pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
    pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    ramp_up_threshold = int(total_epoch*ramp_up)
    if epoch < ramp_up_threshold:
        beta_current = epoch / ramp_up_threshold
        loss = beta_current*nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label.view(b, h, w).long()) + (1 - beta_current)*nn.CrossEntropyLoss(reduction='mean')(pred_norm_prob, label.view(b, h, w).long())
        kld_loss = alpha * (epoch / ramp_up_threshold) * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    else:
        loss = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label.view(b, h, w).long())
        kld_loss = alpha*torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    return loss, kld_loss


def dice_loss(input, target):
    """ This is a normal dice loss function for binary segmentation.

    Args:
        input: output of the segmentation network
        target: ground truth label

    Returns:
        dice score

    """
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


