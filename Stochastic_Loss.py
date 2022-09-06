import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2
torch.backends.cudnn.deterministic = True
# =======================================


def stochastic_noisy_label_loss(pred, cm, mu, logvar, labels, epoch, total_epoch, data, ramp_up=0.5, beta=1.0, lossmode='anneal'):
    """Loss function

    Args:
        pred (torch.Tensor): prediction from semgnetaion network of our model
        cm (torch.Tensor): generated confusion matrix from annotation network
        mu (Torch.Tensor): mean from VAE
        logvar (torch.Tensor): log variance from VAE
        labels (torch.Tensor): noisy labels
        epoch (int): current epoch
        total_epoch (int): total epochs train model on
        data (str): dataset in use. 'mnist' for MNIST data, 'lidc' for LIDC data
        ramp_up (float, optional): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value. Defaults to 0.5.
        beta (float, optional): KL-Divergence constant from beta-vae. Defaults to 1.0.
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling. Defaults to 'anneal'.

    Returns:
        loss (): _description_
        kld_loss (): 
    """
    b, c, h, w = pred.size()    # b: batch size, c: class number, h: height, w: width
    
    # choose random single label form set of annotators
    if data == 'mnist':
        label = random.choice(labels)
    elif data == 'lidc':
        label = labels[:,:,:,:,np.random.choice(labels.shape[4])]

    # b*c x h*w ---> b*h*w x c x 1, p: probabilities from the segmentation network
    pred_noisy = pred.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    # b x c**2 x h x w ---> b*h*w x c x c, reshape cm
    anti_corrpution_cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

    # normalise confusion matrix along the rows:
    anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)

    # matrix multiplication to calculate the predicted noisy segmentation:
    # cm: b*h*w x c x c
    # pred_clean: b*h*w x c x 1
    pred_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b*h*w, c)
    pred_clean = pred_clean.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    # calcuate most likely class for each pixel
    _, pseudo_labels = torch.max(pred, dim=1)

    # Compute KL-Divergence
    # alpha term constant from cyclic annealing, between 0 and 1
    # beta, constant from beta-vae, "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
    # logvar.exp() to recover var
    if lossmode=='cyc':
        alphas = frange_cycle_linear(total_epoch)
        alpha = alphas[epoch]

        kld_loss = alpha * beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    elif lossmode=='anneal':
        # epoch where KL annealing term goes to 1
        ramp_up_threshold = int(total_epoch*ramp_up)

        # gradually increase weight of kl-div loss while epoch less than ramp_up threshold 
        if epoch < ramp_up_threshold:
            beta_current = epoch / ramp_up_threshold
            kld_loss = beta * beta_current * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        else: 
            kld_loss = beta * torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    
    loss_recon1 = nn.CrossEntropyLoss(reduction='mean')(pred_clean, label.view(b, h, w).long()) # loss between reconstructed segmentation and random label
    loss_seg = nn.CrossEntropyLoss(reduction='mean')(pred, label.view(b, h, w).long()) # loss between seg network output and random label
    loss_recon2 =  nn.CrossEntropyLoss(reduction='mean')(pred_clean, pseudo_labels.long())# loss between reconstructd segmentation and seg net output 

    loss = loss_recon1 + loss_seg + kld_loss + loss_recon2
    return loss, kld_loss


def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=5, ratio=0.5):
    """cyclic KL annealing from "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing" - Hao Fu et al.
    Code from: https://github.com/haofuml/cyclical_annealing
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 
