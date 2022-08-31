import os
import errno
import torch
from adamW import AdamW
import timeit
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import pandas as pd

from Utilis import seg_score, generalized_energy_distance
from tensorboardX import SummaryWriter

from Utilis import CustomDataset_punet, calculate_cm, CustomDataset_LIDC, LIDC_collate
from Stochastic_CM import UNet_SCM
from Stochastic_Loss import stochastic_noisy_label_loss
from Utilis import eval_metric

def trainStoch(input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                width,
                depth,
                resolution,
                data_path,
                dataset_tag,
                label_mode=None,
                ramp_up=0.3,
                alpha=20.0,
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
                            resolution=resolution,
                            width=width,
                            depth=depth,
                            latent=6,
                            class_no=class_no,
                            norm='in')

        Exp_name = str(dataset_tag) + '25' + '_width' + str(width) + \
                   '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                   '_repeat' + str(j) + '_e' + str(num_epochs) + \
                   '_lr' + str(learning_rate) + '_save_probability_' + str(save_probability_map) 

        # ====================================================================================================================================================================
        if dataset_tag == 'lidc':
            trainloader, validateloader, testloader, meta_df = getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode)
            
            # ================================
            trainVAE(Stochastic_net,
                            Exp_name,
                            num_epochs,
                            learning_rate,
                            trainloader,
                            validateloader,
                            testloader,
                            ramp_up=ramp_up,
                            alpha=alpha,
                            datatag=dataset_tag,
                            meta_df=meta_df,
                            save_probability_map=save_probability_map)

        elif dataset_tag == 'mnist':
            trainloader, validateloader, testloader = getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode)

            # ================================
            trainVAE(Stochastic_net,
                            Exp_name,
                            num_epochs,
                            learning_rate,
                            trainloader,
                            validateloader,
                            testloader,
                            ramp_up=ramp_up,
                            alpha=alpha,
                            datatag=dataset_tag,
                            save_probability_map=save_probability_map)


def getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode):
    #
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'
    if dataset_tag == 'lidc':
        # get meta file as dataframe
        meta_file = data_path + '/meta/metadata.csv'
        meta_df = pd.read_csv(meta_file)
        # 
        train_dataset = CustomDataset_LIDC(dataset_location=train_path, augmentation=True)
        validate_dataset = CustomDataset_LIDC(dataset_location=validate_path, augmentation=False)
        test_dataset = CustomDataset_LIDC(dataset_location=test_path, augmentation=False)
        # 
        # Dataloaders with custom collate_fn for LIDC data
        trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=5, drop_last=True, collate_fn=LIDC_collate)
        validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False, collate_fn=LIDC_collate)
        testloader = data.DataLoader(test_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False, collate_fn=LIDC_collate)
        #
        return trainloader, validateloader, testloader, meta_df
    #
    elif dataset_tag == 'mnist':
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


def trainVAE(model,
                     model_name,
                     num_epochs,
                     learning_rate,
                     trainloader,
                     validateloader,
                     testloader,
                     ramp_up,
                     alpha,
                     datatag,
                     meta_df=None,
                     save_probability_map=False):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    save_model_name = model_name
    #
    saved_information_path = './99/' + str(datatag)
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
    saved_cm_path = save_path_visual_result + '/progressive_cms'
    try:
        os.mkdir(saved_cm_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_img_path = save_path_visual_result + '/progressive_ims'
    try:
        os.mkdir(saved_img_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    if datatag == 'lidc':
        # data paths to split results depending on ease of detection by annotators
        save_path_subtlety = save_path_visual_result + '/subtlety_'
        #
        for i in range(1, 6):
            try:
                os.mkdir(save_path_subtlety + str(i))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise       
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter('./99/' + str(datatag) + '/Log/Log_' + model_name)

    model.to(device)
    # model_cm.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
    # optimizer2 = torch.optim.Adam(model_cm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    if datatag == 'mnist':
        for epoch in range(num_epochs):
            av_loss, av_kld, av_dice = train(model, device, trainloader, optimizer, epoch, num_epochs, ramp_up, alpha)
            v_dice, v_kld, v_loss, v_ged = validate_stochastic(validateloader, model, device, epoch, num_epochs, ramp_up, alpha, save_cm_path=saved_cm_path, save_img_path=saved_img_path)
            
            print(
                'Step [{}/{}], '
                'Train loss: {:.4f}, '
                'Train kld: {:.4f}, '
                'Train dice: {:.4f},'
                '\nValidate loss: {:.4f},'
                'Validate kld: {:.4f}, '
                'Validate dice: {:.4f}, '
                'Validate ged: {:.4f}, '.format(epoch + 1, num_epochs,
                                                            av_loss,
                                                            av_kld,
                                                            av_dice,
                                                            v_loss[0],
                                                            v_kld[0],
                                                            v_dice[0],
                                                            v_ged[0]))
            #
            writer.add_scalars('scalars', {'train loss': av_loss,
                                            'train kld': av_kld,
                                            'train dice': av_dice,
                                            'val loss': v_loss[0],
                                            'val kld': v_kld[0],
                                            'val dice': v_dice[0],
                                            'val ged': v_ged[0]}, epoch + 1)
                #
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #
        t_dice, t_kld, t_loss, t_ged = test_stochastic(testloader, model, device, epoch, num_epochs, ramp_up, alpha, save_path=save_path_visual_result)
    elif datatag == 'lidc':
        for epoch in range(num_epochs):
            av_loss, av_kld, av_dice = train_lidc(model, device, trainloader, optimizer, epoch, num_epochs, ramp_up, alpha)
            v_dice, v_kld, v_loss, v_ged = validate_stochastic_lidc(validateloader, model, device, epoch, num_epochs, ramp_up, alpha, save_cm_path=saved_cm_path, save_img_path=saved_img_path)
            
            print(
                'Step [{}/{}], '
                'Train loss: {:.4f}, '
                'Train kld: {:.4f}, '
                'Train dice: {:.4f},'.format(epoch + 1, num_epochs,
                                                            av_loss,
                                                            av_kld,
                                                            av_dice))
            #
            writer.add_scalars('scalars', {'train loss': av_loss,
                                            'train kld': av_kld,
                                            'train dice': av_dice}, epoch + 1)
                #
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #
        t_dice, t_kld, t_loss, t_ged = test_stochastic_lidc(testloader, model, device, epoch, num_epochs, ramp_up, alpha, save_path=save_path_subtlety, meta_df=meta_df)

        # if save_probability_map is True:
        #     for class_index in range(c):
        #         #
        #         if c > 0:
        #             v_outputs_logits = v_outputs_logits[:, class_index, :, :]
        #             save_name = save_path_visual_result + '/test_' + str(i) + '_class_' + str(class_index) + '_seg_probability.png'
        #             plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #     #
        # cm_mse = 0
        # #
        # for j, cm in enumerate(v_outputs_logits_noisy):
        #     #
        #     cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
        #     cm = cm / cm.sum(1, keepdim=True)
        #     #
        #     if j < len(cm_all_true):
        #         #
        #         cm_pred_ = cm.sum(0) / (b*h*w)
        #         #
        #         # print(np.shape(cm_pred_))
        #         #
        #         cm_pred_ = cm_pred_.cpu().detach().numpy()
        #         #
        #         # print(np.shape(cm_pred_))
        #         #
        #         cm_true_ = cm_all_true[j]
        #         #
        #         # print(np.shape(cm_true_))
        #         #
        #         cm_mse_each_label = cm_pred_ - cm_true_
        #         #
        #         cm_mse_each_label = cm_mse_each_label**2
        #         # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2
        #         cm_mse += cm_mse_each_label.mean()
        #         #
        #         # print(cm_mse)
        #     #
        #     # v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
        #     # v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        #     # _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
        #     # print('noisy ' + str(nnn) + ' of test ' + str(i))
        #     # print(torch.sum(cm, dim=0) / (b * h * w))
        #     save_name = save_path_visual_result + '/test_' + imagename[0] + '_' + str(i) + '_noisy_' + str(j) + '_seg.png'
        #     #
        #     save_cm_name = save_path_visual_result + '/' + imagename[0] + '_cm.npy'
        #     np.save(save_cm_name, cm.cpu().detach().numpy())
        #     #
        #     # plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')

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
    result_dictionary = {'Test Dice': str(t_dice[0]) + '±' + str(t_dice[1]), 'GED': str(t_ged[0]) + '±' + str(t_ged[1])}
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
        # labels_all.append(labels_wrong)
        # labels_all.append(labels_good)

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

def train_lidc(model, device, train_loader, optimizer, epoch, num_epochs, ramp_up, alpha):
    model.train()

    running_loss = 0
    running_kld_loss = 0
    running_iou = 0
    num_batches = len(train_loader)


    for j, (images, true_image, annots, imagename) in enumerate(train_loader):
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
        outputs_logits, sampled_cm, mean, logvar = model(images)
        # outputs = 0.9*outputs + 0.1*outputs_logits

        # calculate loss:
        seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up, alpha=alpha)
        # calculate the gradients:
        loss = seg_loss + kldloss
        loss.backward()
        # update weights in model:
        optimizer.step()

        train_output = calcPred(outputs_logits, sampled_cm)

        train_iou = seg_score(true_image.cpu().detach().numpy(), train_output.cpu().detach().numpy())
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

# def cm_mse_fn(labels, label_good, cm_index, shape):
#     b, c, h, w = shape
#     cm_all_true = [calculate_cm(pred=label, true=label_good) for label in labels]
#     cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
#     cm = cm / cm.sum(1, keepdim=True)
#     #
#     if cm_index < len(cm_all_true):
#         #
#         cm_pred_ = cm.sum(0) / (b*h*w)
#         #
#         # print(np.shape(cm_pred_))
#         #
#         cm_pred_ = cm_pred_.cpu().detach().numpy()
#         #
#         # print(np.shape(cm_pred_))
#         #
#         cm_true_ = cm_all_true[cm_index]
#         #
#         # print(np.shape(cm_true_))
#         #
#         cm_mse_each_label = cm_pred_ - cm_true_
#         #
#         cm_mse_each_label = cm_mse_each_label**2
#         # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2

#     return cm_mse_each_label.mean()


def validate_stochastic_lidc(data_loader, model, device, epoch, num_epochs, ramp_up, alpha, save_cm_path, save_img_path):
    model.eval()
    
    dice_list = []
    kld_list = []
    seg_loss_list = []
    ged_list = []
    with torch.no_grad():
        for j, (images, true_image, annots, imagename) in enumerate(data_loader):
            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            annots = annots.to(device=device, dtype=torch.float32)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits_original, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            shape = outputs_logits_original.size()
            b, c, h ,w = shape
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up, alpha=alpha)

            output = calcPred(outputs_logits_original, sampled_cm)

            dice = seg_score(true_image.numpy(), output.numpy())
            noisy_labels = [annots[:,:,:,:,annot_idx].reshape(h, w).numpy() for annot_idx in range(annots.shape[4])]

            outputs_logits_original = outputs_logits_original.reshape(b, c, h*w)
            outputs_logits_original = outputs_logits_original.permute(0, 2, 1).contiguous()
            outputs_logits_original = outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)

            noisy_segs = getSamples(outputs_logits_original, model, shape, sample_no=len(noisy_labels))

            if len(noisy_labels) > 1:
                ged = generalized_energy_distance(noisy_labels, noisy_segs, class_no=2)
                ged_list.append(ged)

            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            

            if epoch % 10 == 0:
                if j == 1:
                    # list of images to save
                    images_save = [images[:, 0, :, :].reshape(h, w).cpu().detach().numpy(), true_image.reshape(h, w).cpu().detach().numpy()] + noisy_segs
                    save_name = save_img_path + '/test_' + str(epoch) + '.png'
                    #
                    plt.imsave(save_name, np.hstack(images_save), cmap='gray')

                # reshape cm: [ b , c , c , h , w]= > [ b∗h∗w, c , c ]
                boo = sampled_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
                # boo = torch.softmax(boo, dim=1)
                
                save_path = save_cm_path + '/cm_' + str(epoch) + '.npy'
                np.save(save_path, boo.numpy())

    
    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met

def validate_stochastic(data_loader, model, device, epoch, num_epochs, ramp_up, alpha, save_cm_path, save_img_path):
    model.eval()
    
    dice_list = []
    kld_list = []
    seg_loss_list = []
    ged_list = []
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
            # labels_all.append(labels_wrong)
            # labels_all.append(labels_good)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits_original, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            shape = outputs_logits_original.size()
            b, c, h ,w = shape
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, alpha=alpha)

            output = calcPred(outputs_logits_original, sampled_cm)

            dice = seg_score(labels_good.numpy(), output.numpy())
            noisy_labels = [labels_good.reshape(h,w).numpy(), labels_over.reshape(h, w).numpy(), labels_under.reshape(h, w).numpy(), labels_wrong.reshape(h, w).numpy()]

            outputs_logits_original = outputs_logits_original.reshape(b, c, h*w)
            outputs_logits_original = outputs_logits_original.permute(0, 2, 1).contiguous()
            outputs_logits_original = outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)

            noisy_segs = getSamples(outputs_logits_original, model, shape, sample_no=len(noisy_labels))

            ged = generalized_energy_distance(noisy_labels, noisy_segs[1:], class_no=2)

            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            ged_list.append(ged)

            # if epoch % 10 == 0:
            #     if j == 1:
            #         # list of images to save
            #         images_save = [images[:, 1, :, :].reshape(h, w), labels_good.reshape(h, w).cpu().detach().numpy()] + noisy_segs
            #         save_name = save_img_path + '/test_' + str(epoch) + '.png'
            #         #
            #         plt.imsave(save_name, np.hstack(images_save), cmap='gray')

            #     # reshape cm: [ b , c , c , h , w]= > [ b∗h∗w, c , c ]
            #     boo = sampled_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
            #     # boo = torch.softmax(boo, dim=1)
                
            #     save_path = save_cm_path + '/cm_' + str(epoch) + '.npy'
            #     np.save(save_path, boo.numpy())

    
    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met


def test_stochastic(data_loader, model, device, epoch, num_epochs, ramp_up, alpha, save_path, sample_no=5):
    model.eval()

    dice_list = []
    kld_list = []
    seg_loss_list = []
    ged_list = []
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
            # labels_all.append(labels_wrong)
            # labels_all.append(labels_good)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits_original, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            shape = outputs_logits_original.size()
            b, c, h ,w = shape
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, alpha=alpha)

            output = calcPred(outputs_logits_original, sampled_cm)

            dice = seg_score(labels_good.numpy(), output.numpy())
            noisy_labels = [labels_good.reshape(h, w).numpy(), labels_over.reshape(h, w).numpy(), labels_under.reshape(h, w).numpy(), labels_wrong.reshape(h, w).numpy()]

            outputs_logits_original = outputs_logits_original.reshape(b, c, h*w)
            outputs_logits_original = outputs_logits_original.permute(0, 2, 1).contiguous()
            outputs_logits_original = outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)

            noisy_segs = getSamples(outputs_logits_original, model, shape, sample_no=sample_no)
            
            ged_noisy_segs = noisy_segs[1:len(noisy_labels)+1]
            ged = generalized_energy_distance(noisy_labels, ged_noisy_segs, class_no=2)

            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            ged_list.append(ged)

            # list of images to save
            images_save = [images[:, 1, :, :].reshape(h, w), labels_good.reshape(h, w).cpu().detach().numpy()] + noisy_segs
            save_name = save_path + '/test_' + str(j) + '.png'
            #
            plt.imsave(save_name, np.hstack(images_save), cmap='gray')


    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met


def test_stochastic_lidc(data_loader, model, device, epoch, num_epochs, ramp_up, alpha, save_path, meta_df, sample_no=5):
    model.eval()

    dice_list = []
    kld_list = []
    seg_loss_list = []
    ged_list = []
    with torch.no_grad():
        for j, (images, true_image, annots, imagename) in enumerate(data_loader):
            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            annots = annots.to(device=device, dtype=torch.float32)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits_original, sampled_cm, mean, logvar = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits
            shape = outputs_logits_original.size()
            b, c, h ,w = shape
            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up, alpha=alpha)

            output = calcPred(outputs_logits_original, sampled_cm)

            dice = seg_score(true_image.numpy(), output.numpy())
            noisy_labels = [annots[:,:,:,:,annot_idx].reshape(h, w).numpy() for annot_idx in range(annots.shape[4])]

            outputs_logits_original = outputs_logits_original.reshape(b, c, h*w)
            outputs_logits_original = outputs_logits_original.permute(0, 2, 1).contiguous()
            outputs_logits_original = outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)

            noisy_segs = getSamples(outputs_logits_original, model, shape, sample_no=len(noisy_labels))
            
            if len(noisy_labels) > 1:
                ged = generalized_energy_distance(noisy_labels, noisy_segs, class_no=2)

            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            ged_list.append(ged)


            _, patient_id, _, nod_no, _, _ = imagename[0].split('_')
            subtlety = meta_df.loc[(meta_df['patient_id'] == int(patient_id)) & (meta_df['nodule_no'] == int(nod_no))]['subtlety'].item()

            # list of images to save
            images_save = [images.reshape(h, w).cpu().detach().numpy(), true_image.reshape(h, w).cpu().detach().numpy()] + noisy_segs
            save_name = save_path + str(subtlety) + '/test_' + str(j) + '.png'
            #
            plt.imsave(save_name, np.hstack(images_save), cmap='gray')


    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met


def getSamples(output_logits_original, model, shape, sample_no=5):
    b, c, h, w = shape
    samples = model.cm_network.sample(sample_no)
    _, seg = torch.max(output_logits_original, dim=1)
    sample_list = [seg.reshape(h,w).cpu().detach().numpy()]
    for sample in samples:
        sample_cm = torch.unsqueeze(sample, dim=0)

        anti_corrpution_cm = sample_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        # anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)
        anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)

        # pred_norm_prob_noisy = pred_norm_prob_noisy.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
        outputs_clean = torch.bmm(anti_corrpution_cm, output_logits_original).view(b * h * w, c)
        outputs_clean = outputs_clean.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        # v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        _, v_outputs_logits = torch.max(outputs_clean, dim=1)

        # save_name = save_path_visual_result + '/test_' + str(i) + '_seg_' + str(k) + '.png'
        sample_list.append(v_outputs_logits.reshape(h, w).cpu().detach().numpy())
        # plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')

    return sample_list

