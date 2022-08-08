import os
import errno
from re import I
from wsgiref import validate
import torch
from adamW import AdamW
import timeit
import imageio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from Utilis import evaluate_LIDC, seg_score, generalized_energy_distance, validate_LIDC
from tensorboardX import SummaryWriter

from Utilis import CustomDataset_punet, calculate_cm
from Stochastic_CM import UNet_SCM
from Stochastic_Loss import stochastic_noisy_label_loss

def trainStoch(input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                width,
                depth,
                data_path,
                dataset_tag,
                label_mode,
                ramp_up,
                alpha,
                save_probability_map=True):

    """ This is the panel to control the hyper-parameter of training of our methods.

    Args:
        input_dim: channel number of input image, for example, 3 for RGB
        class_no: number of classes of classification
        repeat: repat the same experiments with different stochastic seeds, we normally run each experiment at least 3 times
        train_batchsize: training batch size, this depends on the GPU memory
        validate_batchsize: we normally set-up as 1
        num_epochs: training epoch length
        learning_rate:
        input_height: resolution of input image
        input_width: resolution of input image
        alpha: regularisation strength hyper-parameter
        width: channel number of first encoder in the segmentation network, for the standard U-net, it is 64
        depth: down-sampling stages of the segmentation network
        data_path: path to where you store your all of your data
        dataset_tag: 'mnist' for MNIST; 'brats' for BRATS 2018; 'lidc' for LIDC lung data set
        label_mode: 'multi' for multi-class of proposed method; 'p_unet' for baseline probabilistic u-net; 'normal' for binary on MNIST; 'binary' for general binary segmentation
        loss_f: 'noisy_label' for our noisy label function, or 'dice' for dice loss
        save_probability_map: if True, we save all of the probability maps of output of networks

    Returns:

    """
    for j in range(1, repeat + 1):
        #
        Stochastic_net = UNet_SCM(in_ch=input_dim,
                            resolution=28,
                            width=width,
                            depth=depth,
                            latent=512,
                            class_no=class_no,
                            norm='in')

        Exp_name = 'Seg_UNet_DCMs_Direct_' + '_width' + str(width) + \
                   '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                   '_repeat' + str(j) + '_e' + str(num_epochs) + \
                   '_lr' + str(learning_rate) + '_save_probability_' + str(save_probability_map) 

        # ====================================================================================================================================================================
        trainloader, validateloader, testloader = getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode)
        # ================================
        trainMNIST(Stochastic_net,
                         Exp_name,
                         num_epochs,
                         learning_rate,
                         trainloader,
                         validateloader,
                         testloader,
                         ramp_up=ramp_up,
                         alpha=alpha,
                         save_probability_map=save_probability_map)


def getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode):
    #
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'
    #
    # prepare data sets using our customdataset
    train_dataset = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=True)
    validate_dataset = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)
    test_dataset = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)

    # putting dataset into data loaders
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    #
    return trainloader, validateloader, testloader

# =====================================================================================================================================


def trainMNIST(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     trainloader,
                     validateloader,
                     testloader,
                     ramp_up,
                     alpha,
                     save_probability_map):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    save_model_name = model_name
    #
    saved_information_path = './Results/MNIST'
    #
    try:
        os.makedirs(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_information_path = saved_information_path + '/Results_' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    save_path_visual_result = saved_information_path + '/visual_results'
    try:
        os.mkdir(save_path_visual_result)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter('./Results/MNIST/Log/Log_' + model_name)

    model.to(device)
    # model_cm.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    # optimizer2 = torch.optim.Adam(model_cm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        av_loss, av_kld, av_dice = train(model, device, trainloader, optimizer, epoch, num_epochs, ramp_up, alpha)
        v_dice, v_kld, v_loss = validate_stochastic(validateloader, model, device, epoch, num_epochs, ramp_up, alpha)
        
        print(
            'Step [{}/{}], '
            'Train loss: {:.4f}, '
            'Train kld: {:.4f}, '
            'Train dice: {:.4f},'
            '\nValidate loss: {:.4f},'
            'Validate kld: {:.4f}, '
            'Validate dice: {:.4f}, '.format(epoch + 1, num_epochs,
                                                        av_loss,
                                                        av_kld,
                                                        av_dice,
                                                        v_loss,
                                                        v_kld,
                                                        v_dice))
        #
        writer.add_scalars('scalars', {'train loss': av_loss,
                                        'train kld': av_kld,
                                        'train dice': av_dice,
                                        'val loss': v_loss,
                                        'val kld': v_kld,
                                        'val dice': v_dice}, epoch + 1)
            #
            # # # ================================================================== #
            # # #                        TensorboardX Logging                        #
            # # # # ================================================================ #

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate*((1 - epoch / num_epochs)**0.999)
        #
    model.eval()
    #
    for i, (v_images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(testloader):
        #
        cm_all_true = []
        #
        cm_over_true = calculate_cm(pred=labels_over, true=labels_good)
        cm_under_true = calculate_cm(pred=labels_under, true=labels_good)
        cm_wrong_true = calculate_cm(pred=labels_wrong, true=labels_good)
        #
        cm_all_true.append(cm_over_true)
        cm_all_true.append(cm_under_true)
        cm_all_true.append(cm_wrong_true)
        #
        # cm_all_true_result = sum(cm_all_true) / len(cm_all_true)
        #
        v_images = v_images.to(device=device, dtype=torch.float32)
        #
        v_outputs_logits_original, v_outputs_logits_noisy, _, _ = model(v_images)
        #
        b, c, h, w = v_outputs_logits_original.size()
        #
        # v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        #
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)
        #
        nnn = 1
        #
        v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h*w)
        v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
        v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)
        #
        samples = model.cm_network.sample(5)
        #
        sample_list = []
        for j, sample in enumerate(samples):
            sample_cm = torch.unsqueeze(sample, dim=0)

            anti_corrpution_cm = sample_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
            # anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)
            anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)

            # pred_norm_prob_noisy = pred_norm_prob_noisy.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
            outputs_clean = torch.bmm(anti_corrpution_cm, v_outputs_logits_original).view(b * h * w, c)
            v_outputs_logits_o = outputs_clean.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

            # v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
            _, v_outputs_logits = torch.max(v_outputs_logits_o, dim=1)

            # save_name = save_path_visual_result + '/test_' + str(i) + '_seg_' + str(k) + '.png'
            sample_list.append(v_outputs_logits.reshape(h, w).cpu().detach().numpy())
            # plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')

        save_name = save_path_visual_result + '/test_' + str(i) + '_seg_samples.png'
        save_name_label = save_path_visual_result + '/test_' + str(i) + '_label.png'
        #
        # bb, cc, hh, ww = v_images.size()
        #
        # for ccc in range(cc):
        #     #
        #     save_name_slice = save_path + '/test_' + imagename[0] + '_' + str(i) + '_slice_' + str(ccc) + '.png'
        #     plt.imsave(save_name_slice, v_images[:, ccc, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        plt.imsave(save_name, np.hstack(sample_list), cmap='gray')
        plt.imsave(save_name_label, labels_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        if save_probability_map is True:
            for class_index in range(c):
                #
                if c > 0:
                    v_outputs_logits = v_outputs_logits_o[:, class_index, :, :]
                    save_name = save_path_visual_result + '/test_' + str(i) + '_class_' + str(class_index) + '_seg_probability.png'
                    plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        cm_mse = 0
        #
        for j, cm in enumerate(v_outputs_logits_noisy):
            #
            cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            #
            if j < len(cm_all_true):
                #
                cm_pred_ = cm.sum(0) / (b*h*w)
                #
                # print(np.shape(cm_pred_))
                #
                cm_pred_ = cm_pred_.cpu().detach().numpy()
                #
                # print(np.shape(cm_pred_))
                #
                cm_true_ = cm_all_true[j]
                #
                # print(np.shape(cm_true_))
                #
                cm_mse_each_label = cm_pred_ - cm_true_
                #
                cm_mse_each_label = cm_mse_each_label**2
                # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2
                cm_mse += cm_mse_each_label.mean()
                #
                # print(cm_mse)
            #
            v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
            v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
    # save model
    stop = timeit.default_timer()
    #
    print('Time: ', stop - start)
    #
    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final_seg.pt'
    #
    path_model = save_model_name_full
    #
    torch.save(model, path_model)
    #
    # save_model_name_full = saved_model_path + '/' + save_model_name + '_Final_cm.pt'
    #
    # path_model = save_model_name_full
    #
    # torch.save(model_cm, path_model)
    #
    result_dictionary = {'Test Dice': str(v_dice), 'Test CM MSE': str(cm_mse / (i + 1))}
    ff_path = saved_information_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    print('\nTraining finished and model saved\n')
    #
    return model

def train(model, device, train_loader, optimizer, epoch, num_epochs, ramp_up, alpha):
    model.train()

    running_loss = 0
    running_kld_loss = 0
    running_iou = 0
    num_batches = len(train_loader)

    for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(train_loader):
        # zero graidents before each iteration
        optimizer.zero_grad()

        # cast numpy data into tensor float
        images = images.to(device=device, dtype=torch.float32)
        labels_over = labels_over.to(device=device, dtype=torch.float32)
        labels_under = labels_under.to(device=device, dtype=torch.float32)
        labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
        labels_good = labels_good.to(device=device, dtype=torch.float32)

        labels_all = []
        labels_all.append(labels_over)
        labels_all.append(labels_under)
        labels_all.append(labels_wrong)
        labels_all.append(labels_good)

        # model has two outputs:
        # first one is the probability map for true ground truth
        # second one is a list collection of probability maps for different noisy ground truths
        outputs_logits, sampled_cm, mean, logvar = model(images)
        # outputs = 0.9*outputs + 0.1*outputs_logits
        
        # calculate loss:
        seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, alpha=alpha)
        # print(images.size())
        loss = seg_loss + kldloss
        # calculate the gradients:
        # loss = lossss(model, sampled_outputs_logits, stochastic_cm, mean, logvar, label, epoch, num_epochs, ramp_up, alpha)
        loss.backward()
        # update weights in model:
        optimizer.step()

        train_output = calcPred(outputs_logits, sampled_cm)

        train_iou = seg_score(labels_good.cpu().detach().numpy(), train_output.cpu().detach().numpy())
        running_loss += seg_loss
        running_kld_loss += kldloss
        running_iou += train_iou

        # if (j + 1) == 1:
        #     print(
        #         'Step [{}/{}], '
        #         'Train loss: {:.4f}, '
        #         'Train kld: {:.4f},'
        #         'Train dice: {:.4f},'.format(epoch + 1, num_epochs,
        #                                                     running_loss / (j + 1),
        #                                                     running_kld_loss / (j + 1),
        #                                                     running_iou / (j + 1)))
    av_loss = running_loss / num_batches
    av_kld = running_kld_loss / num_batches
    av_dice = running_iou / num_batches

    return av_loss, av_kld, av_dice

def calcPred(pred_seg_logits, cm):
    # Now outputs_logits is the noisy seg:
    b_, c_, h_, w_ = pred_seg_logits.size()

    pred_noisy = pred_seg_logits.view(b_, c_, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_, 1)

    # reshape cm: [ b , c , c , h , w]= > [ b∗h∗w, c , c ]
    anti_corrpution_cm = cm.view(b_, c_ ** 2, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_ * c_).view(b_ * h_ * w_, c_, c_)
    anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)
    # compute estimated annotators noisy seg proability
    outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b_ * h_ * w_, c_)
    # reshape: [ b∗h∗w, c , 1 ] => [ b , c , h , w ]
    outputs_clean = outputs_clean.view(b_, h_ * w_, c_).permute(0, 2, 1).contiguous().view(b_, c_, h_, w_)

    # probability -> binary mask
    _, train_output = torch.max(outputs_clean, dim=1)

    return train_output

def validate_stochastic(data_loader, model, device, epoch, num_epochs, ramp_up, alpha):
    model.eval()

    running_dice = 0
    running_kld_loss = 0
    running_seg_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(data_loader):

            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            labels_over = labels_over.to(device=device, dtype=torch.float32)
            labels_under = labels_under.to(device=device, dtype=torch.float32)
            labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
            labels_good = labels_good.to(device=device, dtype=torch.float32)

            labels_all = []
            labels_all.append(labels_over)
            labels_all.append(labels_under)
            labels_all.append(labels_wrong)
            labels_all.append(labels_good)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, alpha=alpha)

            train_output = calcPred(outputs_logits, sampled_cm)
            train_iou = seg_score(labels_good.numpy(), train_output.numpy())
            
            running_seg_loss += seg_loss
            running_kld_loss += kldloss
            running_dice += train_iou
    
    av_dice = running_dice / num_batches
    av_kld = running_kld_loss / num_batches
    av_seg_loss = running_seg_loss / num_batches

    return av_dice, av_kld, av_seg_loss

def test(data_loader, model, device, epoch, num_epochs, ramp_up, alpha):
    model.eval()

    running_dice = 0
    running_kld_loss = 0
    running_seg_loss = 0
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(data_loader):

            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            labels_over = labels_over.to(device=device, dtype=torch.float32)
            labels_under = labels_under.to(device=device, dtype=torch.float32)
            labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
            labels_good = labels_good.to(device=device, dtype=torch.float32)

            labels_all = []
            labels_all.append(labels_over)
            labels_all.append(labels_under)
            labels_all.append(labels_wrong)
            labels_all.append(labels_good)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits.detach(), sampled_cm.detatch(), mean.detatch(), logvar.detatch(), labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, alpha=alpha)

            train_output = calcPred(outputs_logits, sampled_cm)
            train_iou = seg_score(labels_good.detach().numpy(), train_output.detach().numpy())
            
            running_seg_loss += seg_loss
            running_kld_loss += kldloss
            running_dice += train_iou
    
    av_dice = running_dice / num_batches
    av_kld = running_kld_loss / num_batches
    av_seg_loss = running_seg_loss / num_batches

    return av_dice, av_kld, av_seg_loss

