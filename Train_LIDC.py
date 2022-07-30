import os
import errno
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
from torch.autograd import Variable

from Utilis import CustomDataset_LIDC, calculate_cm, LIDC_collate
from Deterministic_LIDC_CM import UNet_DCM
from Deterministic_LIDC_Loss import deterministic_noisy_label_loss

from Utilis import evaluate_noisy_label_4, evaluate_noisy_label_5, evaluate_noisy_label_6

import pandas as pd

def trainModels(input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                width,
                depth,
                data_path,
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
        Segmentation_net = UNet_DCM(in_ch=input_dim,
                            resolution=64,
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
        trainloader, validateloader, testloader, data_length, meta_df = getData(train_batchsize, validate_batchsize, data_path)
        # ================================
        trainLIDC(Segmentation_net,
                         Exp_name,
                         num_epochs,
                         data_length,
                         learning_rate,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         meta_df,
                         class_no=class_no,
                         save_probability_map=save_probability_map)


def getData(train_batchsize, validate_batchsize, data_path):
    #
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'
    # get meta file as dataframe
    meta_file = data_path + '/meta/metadata.csv'
    meta_df = pd.read_csv(meta_file)
    #
    train_dataset = CustomDataset_LIDC(dataset_location=train_path, augmentation=True)
    #
    validate_dataset = CustomDataset_LIDC(dataset_location=validate_path, augmentation=False)
    #
    test_dataset = CustomDataset_LIDC(dataset_location=test_path, augmentation=False)
    #
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=5, drop_last=True, collate_fn=LIDC_collate)
    #
    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False, collate_fn=LIDC_collate)
    #
    testloader = data.DataLoader(test_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False, collate_fn=LIDC_collate)
    #
    return trainloader, validateloader, testloader, len(train_dataset), meta_df

# =====================================================================================================================================


def trainLIDC(model_seg,
                     model_name,
                     num_epochs,
                     data_length,
                     learning_rate,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     meta_df,
                     class_no,
                     save_probability_map):
    #
    # change log names
    iteration_amount = data_length // train_batchsize - 1
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    save_model_name = model_name
    #
    saved_information_path = './Results/LIDC'
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
    save_path_subtlety = saved_information_path + '/subtlety_'
    #
    for i in range(1, 6):
        try:
            os.mkdir(save_path_subtlety + str(i))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
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
    writer = SummaryWriter('./Results/LIDC/Log/Log_' + model_name)

    model_seg.to(device)
    # model_cm.to(device)

    optimizer = AdamW(model_seg.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-5)
    # optimizer2 = torch.optim.Adam(model_cm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        #
        model_seg.train()
        # model_cm.train()
        running_loss = 0
        # running_loss_ce = 0
        # running_loss_trace = 0
        running_iou = 0
        #
        for j, (images, true_image, annots, imagename) in enumerate(trainloader):
            # b, c, h, w = images.size()

            # zero graidents before each iteration
            optimizer.zero_grad()

            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            annots = annots.to(device=device, dtype=torch.float32)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits, stochastic_cm = model_seg(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits

            # calculate loss:
            loss = deterministic_noisy_label_loss(outputs_logits, stochastic_cm, annots, epoch, num_epochs)
            # calculate the gradients:
            loss.backward()
            # update weights in model:
            optimizer.step()

            # Now outputs_logits is the noisy seg:
            b_, c_, h_, w_ = outputs_logits.size()
            # pred_norm_prob_noisy = nn.Softmax(dim=1)(outputs_logits)
            pred_noisy = outputs_logits.view(b_, c_, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_, 1)
            anti_corrpution_cm = stochastic_cm.view(b_, c_ ** 2, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_ * c_).view(b_ * h_ * w_, c_, c_)
            anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)
            outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b_ * h_ * w_, c_)
            outputs_clean = outputs_clean.view(b_, h_ * w_, c_).permute(0, 2, 1).contiguous().view(b_, c_, h_, w_)

            _, train_output = torch.max(outputs_clean, dim=1)
            train_iou = seg_score(true_image.cpu().detach().numpy(), train_output.cpu().detach().numpy())
            running_loss += loss
            # running_loss_ce += loss_ce
            # running_loss_trace += loss_trace
            running_iou += train_iou
            #
            # if (j + 1) % iteration_amount == 0:
            if (j + 1) == 1:
                #
                v_dice = validate_LIDC(data_loader=validateloader,
                                                        model1=model_seg,
                                                        device=device)
                #
                print(
                    'Step [{}/{}], '
                    'Train loss: {:.4f}, '
                    'Train dice: {:.4f},'
                    'Validate dice: {:.4f},'.format(epoch + 1, num_epochs,
                                                                running_loss / (j + 1),
                                                                running_iou / (j + 1),
                                                                v_dice))
                #
                writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                                'train iou': running_iou / (j + 1),
                                                'val iou': v_dice}, epoch + 1)
                #
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate*((1 - epoch / num_epochs)**0.999)
        #
    model_seg.eval()
    # model_cm.eval()
    # save_path = './Exp_Results_Noisy_labels'
    # #
    # try:
    #     #
    #     os.mkdir(save_path)
    #     #
    # except OSError as exc:
    #     #
    #     if exc.errno != errno.EEXIST:
    #         #
    #         raise
    #     #
    #     pass
    # #
    # save_path = './Exp_Results_Noisy_labels/LIDC' 
    # #
    # try:
    #     #
    #     os.mkdir(save_path)
    #     #
    # except OSError as exc:
    #     #
    #     if exc.errno != errno.EEXIST:
    #         #
    #         raise
    #     #
    #     pass
    # #
    # save_path = save_path + '/Exp_' + \
    #             '_Noisy_Label_Net_' + save_model_name
    # #
    # try:
    #     #
    #     os.mkdir(save_path)
    #     #
    # except OSError as exc:
    #     #
    #     if exc.errno != errno.EEXIST:
    #         #
    #         raise
    #     #
    #     pass
    #
    for i, (t_images, t_true_images, t_annots, t_imagename) in enumerate(testdata):
        #
        cm_mse = 0
        #
        t_images = t_images.to(device=device, dtype=torch.float32)
        #
        v_outputs_logits_original, stochastic_cm = model_seg(t_images)
        #
        b, c, h, w = v_outputs_logits_original.size()
        #
        v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        #
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)
        #
        # for k, name in enumerate(t_imagename):
        #     _, patient_id, _, nod_no, _, slice_no = name.split('_')

        #     subtlety = meta_df.loc[(meta_df['patient_id'] == int(patient_id)) & (meta_df['nodule_no'] == int(nod_no))]['subtlety'].item()
            
        #     save_name = save_path_subtlety + str(subtlety) + '/test_' + name + '_seg.png'
        #     save_name_label = save_path_subtlety + str(subtlety) + '/test_' + name + '_label.png'
        #     save_name_scan = save_path_subtlety + str(subtlety) + '/test_' + name + '_img.png'

        #     plt.imsave(save_name_scan, t_images[k].reshape(h, w).cpu().detach().numpy(), cmap='gray')
            
        #     if save_probability_map is True:
        #             save_name = save_path + '/test_' + name + '_seg_probability.png'
        #             plt.imsave(save_name, v_outputs_logits[k].reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h*w)
        v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
        v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)
        #

        #
        for j, (name, cm) in enumerate(zip(t_imagename, stochastic_cm)):
            ## imagename part
            _, patient_id, _, nod_no, _, _ = name.split('_')

            subtlety = meta_df.loc[(meta_df['patient_id'] == int(patient_id)) & (meta_df['nodule_no'] == int(nod_no))]['subtlety'].item()
            
            # save_name = save_path_subtlety + str(subtlety) + '/test_' + name + '_seg.png'
            save_name_label = save_path_subtlety + str(subtlety) + '/test_' + name + '_label.png'
            save_name_scan = save_path_subtlety + str(subtlety) + '/test_' + name + '_img.png'

            plt.imsave(save_name_scan, t_images[j].reshape(h, w).cpu().detach().numpy(), cmap='gray')
            plt.imsave(save_name_label, true_image[j].reshape(h, w).cpu().detach().numpy(), cmap='gray')

            if save_probability_map is True:
                    save_name = save_path_subtlety + str(subtlety) + '/test_' + name + '_seg_probability.png'
                    plt.imsave(save_name, v_outputs_logits[j].reshape(h, w).cpu().detach().numpy(), cmap='gray')

            ## cm part
            cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            #
            cm = cm / cm.sum(1, keepdim=True)
            #
            v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
            #
            v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            rand_annot = t_annots[:,:,:,:,np.random.choice(t_annots.shape[4])]
            cm_true_ = calculate_cm(pred=rand_annot, true=t_true_images)
            #
            cm_pred_ = cm.sum(0) / (b*h*w)
            #
            # print(np.shape(cm_pred_))
            #
            cm_pred_ = cm_pred_.cpu().detach().numpy()
            #
            cm_mse_each_label = cm_pred_ - cm_true_
            #
            cm_mse_each_label = cm_mse_each_label**2
            # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2
            cm_mse += cm_mse_each_label.mean()
            #
            # print(cm_mse)
            #
            _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
            #
            # print('noisy ' + str(nnn) + ' of test ' + str(i))
            # print(torch.sum(cm, dim=0) / (b * h * w))
            # nnn += 1
            # print('\n')
            #
            save_name = save_path_subtlety + str(subtlety) + '/test_' + name + '_seg.png'
            #
            save_cm_name = save_path_subtlety + str(subtlety) + '/' + name + '_cm.npy'
            np.save(save_cm_name, cm.cpu().detach().numpy())
            #
            plt.imsave(save_name, v_noisy_output[j].reshape(h, w).cpu().detach().numpy(), cmap='gray')
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
    torch.save(model_seg, path_model)
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
    return model_seg

