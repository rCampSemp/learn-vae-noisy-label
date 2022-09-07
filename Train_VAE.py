import os
import errno
import torch
from adamW import AdamW
import timeit
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import pandas as pd

from tensorboardX import SummaryWriter

from Utilis import CustomDataset_punet, CustomDataset_LIDC, LIDC_collate
from Stochastic_CM import UNet_SCM

# mnist helper funcs
from Utilis import train, validate_stochastic, test_stochastic

# lidc helper funcs
from Utilis import train_lidc, validate_stochastic_lidc, test_stochastic_lidc

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
                beta=20.0,
                lossmode='anneal'):

    """ Script for training our proposed model

    Args:
        input_dim (int): channel number of input image, for example, 3 for RGB
        class_no (int): number of classes of classification
        repeat (int): repat the same experiments with different stochastic seeds, we normally run each experiment at least 3 times
        train_batchsize (int): training batch size, this depends on the GPU memory
        validate_batchsize (int): we normally set-up as 1
        num_epochs (int): training epoch length
        learning_rate (float):
        width (int): channel number of first encoder in the segmentation network
        depth (int): down-sampling stages of the segmentation network
        resolution (int): size of input image, i.e. array size resolution x resolution 
        data_path (str): path to where you store your all of your data
        dataset_tag (str): 'mnist' for MNIST; 'lidc' for LIDC lung data set
        label_mode (str): 'multi' for multi-class of proposed method; 'p_unet' for baseline probabilistic u-net
        ramp_up (float, optional): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value. Defaults to 0.3
        beta (float, optional): KL-Divergence constant from beta-vae. Defaults to 20.0
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling. Defaults to 'anneal'.
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

        Exp_name = str(dataset_tag) + '10' + '_width' + str(width) + \
                   '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                   '_repeat' + str(j) + '_e' + str(num_epochs) + \
                   '_lr' + str(learning_rate) + '_lossmode_' + str(lossmode)

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
                            beta=beta,
                            datatag=dataset_tag,
                            meta_df=meta_df,
                            lossmode='anneal')

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
                            beta=beta,
                            datatag=dataset_tag,
                            lossmode='anneal')


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
                     beta,
                     datatag,
                     meta_df=None,
                     lossmode='anneal'):
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    save_model_name = model_name
    #
    saved_information_path = './15/' + str(datatag)
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
    writer = SummaryWriter('./15/' + str(datatag) + '/Log/Log_' + model_name)

    model.to(device)
    # model_cm.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

    start = timeit.default_timer()

    if datatag == 'mnist':
        for epoch in range(num_epochs):
            av_loss, av_kld, av_dice = train(model, device, trainloader, optimizer, epoch, num_epochs, beta, ramp_up, lossmode)
            v_dice, v_kld, v_loss, v_ged = validate_stochastic(validateloader, model, device, epoch, num_epochs, ramp_up, beta, save_cm_path=saved_cm_path, save_img_path=saved_img_path, lossmode=lossmode)
            
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
        t_dice, t_ged = test_stochastic(testloader, model, device, save_path=save_path_visual_result)
    elif datatag == 'lidc':
        for epoch in range(num_epochs):
            av_loss, av_kld, av_dice = train_lidc(model, device, trainloader, optimizer, epoch, num_epochs, ramp_up, beta, lossmode)
            v_dice, v_kld, v_loss, v_ged = validate_stochastic_lidc(validateloader, model, device, epoch, num_epochs, ramp_up, beta, save_cm_path=saved_cm_path, save_img_path=saved_img_path, lossmode=lossmode)
            
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
        t_dice, t_ged = test_stochastic_lidc(testloader, model, device, save_path=save_path_subtlety, meta_df=meta_df)

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













