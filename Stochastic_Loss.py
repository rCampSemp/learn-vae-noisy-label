import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def stochastic_noisy_label_loss(pred, cm, mu, logvar, labels, epoch, total_epoch, ramp_up=0.5, alpha=1.0):
    """
    Under construction
    """
    # regularisation = 0.0
    b, c, h, w = pred.size()

    # # self attention on pred norm:
    # pred_theta = pred.view(b, c, h * w)
    # pred_phi = pred.view(b, c, h * w).permute(0, 2, 1).contiguous()
    # score = F.softmax(torch.bmm(pred_theta, pred_phi), dim=-1)
    # pred = torch.bmm(score, pred_theta)
    # pred_norm_prob = nn.Softmax(dim=1)(pred.view(b, c, h, w))

    # without calibration
    # pred_norm_prob_noisy = nn.Softmax(dim=1)(pred)

    # b*c x h*w ---> b*h*w x c x 1
    pred_noisy = pred.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
    # label_noisy: noisy label, b x h x w
    # convex combination of noisy labels:
    weights = np.random.dirichlet((1, 1, 1, 1), size=1)
    for i, each_label in enumerate(labels):
        if i == 0:
            label = each_label*weights[0][i]
        else:
            label += each_label*weights[0][i]
    label = (label > 0.5).float()
    # random choice:
    # label = random.choice(labels)

    # loss = 0
    # kld_loss = 0
    # b x c**2 x h x w ---> b*h*w x c x c
    anti_corrpution_cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # normalisation along the rows:
    anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)
    # anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)

    # matrix multiplication to calculate the predicted noisy segmentation:
    # cm: b*h*w x c x c
    # pred_clean: b*h*w x c x 1
    pred_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b*h*w, c)
    pred_clean = pred_clean.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    _, pseudo_labels = torch.max(pred, dim=1)
    ramp_up_threshold = int(total_epoch*ramp_up)

    # loss = nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) + nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())
    # loss += 1*torch.trace(torch.transpose(torch.sum(anti_corrpution_cm, dim=0), 0, 1)).sum() / (b * h * w)
    # kld_loss = alpha*torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    # kld_loss = 0

    if epoch < ramp_up_threshold:
        beta_current = epoch / ramp_up_threshold
        # loss = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label.view(b, h, w).long())
        # print(pred_norm.size())
        loss = beta_current*nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) + (1 - beta_current)*nn.CrossEntropyLoss(reduction='mean')(pred_clean, label.view(b, h, w).long()) + nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())
        kld_loss = alpha * (epoch / ramp_up_threshold) * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    else:
        loss = nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) + nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())
        kld_loss = alpha*torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    return loss, kld_loss


# def global_stochastic_noisy_label_loss(pred, cm, mu, logvar, labels, epoch, total_epoch, ramp_up=0.5, alpha=1.0):
#     # regularisation = 0.0
#     b, c, h, w = pred.size()
#
#     # normalise the segmentation output tensor along dimension 1
#     pred_norm_prob = nn.Softmax(dim=1)(pred)
#
#     # b x c x h x w ---> b x hw x c
#     pred_norm = pred_norm_prob.view(b, c, h*w).permute(0, 2, 1).contiguous()
#
#     # cm: learnt confusion matrix for each noisy label, b x c**2
#     # label_noisy: noisy label, b x h x w
#     # convex combination of noisy labels:
#     weights = np.random.dirichlet((1, 1, 1, 1), size=1)
#     for i, each_label in enumerate(labels):
#         if i == 0:
#             label = each_label*weights[0][i]
#         else:
#             label += each_label*weights[0][i]
#     label = (label > 0.5).float()
#     # random choice:
#     # label = random.choice(labels)
#
#     # b x c**2 ---> b x c x c
#     cm = cm.view(b, c ** 2).view(b, c, c)
#
#     # normalisation along the rows:
#     # cm = cm / cm.sum(1, keepdim=True)
#     cm = torch.softmax(cm, dim=1)
#
#     # matrix multiplication to calculate the predicted noisy segmentation:
#     # cm: b x c x c
#     # pred_noisy: b x h*w x c
#     pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
#     pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
#
#     ramp_up_threshold = int(total_epoch*ramp_up)
#     if epoch < ramp_up_threshold:
#         beta_current = epoch / ramp_up_threshold
#         loss = beta_current*nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label.view(b, h, w).long()) + (1 - beta_current)*nn.CrossEntropyLoss(reduction='mean')(pred_norm_prob, label.view(b, h, w).long())
#         kld_loss = alpha * (epoch / ramp_up_threshold) * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
#     else:
#         loss = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label.view(b, h, w).long())
#         kld_loss = alpha*torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
#
#     return loss, kld_loss


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


