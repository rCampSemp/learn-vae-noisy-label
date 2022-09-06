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

from Utilis import CustomDataset_punet, CustomDataset_LIDC, LIDC_collate
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

def train(model, device, train_loader, optimizer, epoch, num_epochs, beta, ramp_up=None, lossmode='anneal'):
    """Training regime at one step of training for MNIST dataset

    Args:
        model (nn.Module): UNet_SCM our proposed model
        device (torch.device): device type responsible to load tensors into memory
        train_loader (torch.utils.data.DataLoader): iterable over MNIST training dataset
        optimizer (torch.optim.optimizer): PyTorch optimizer
        epoch (int): current epoch 
        num_epochs (int): total number of epochs to train model 
        beta (int or float): KL-Divergence constant from beta-vae
        ramp_up (float, optional): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value.  Defaults to None.
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling. Defaults to 'anneal'.
        
    Returns:
        av_loss (torch.Tensor): 1-element tensor containing average loss from training loop
        av_kld (torch.Tensor): 1-element tensor containing average KL-divergence from training loop 
        av_dice (torch.Tensor): 1-element tensor containing average DICE score from training loop
    """
    model.train()

    # initialize training metrics
    running_loss = 0
    running_kld_loss = 0
    running_dice = 0
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

        # collect all noisy labels
        labels_all = []
        labels_all.append(labels_over)
        labels_all.append(labels_under)
        labels_all.append(labels_wrong)
        labels_all.append(labels_good)

        # model has four outputs:
        # 1: the probability map for true ground truth
        # 2: probability map for noisy ground truth
        # 3: mean of the latent space
        # 4: log variance (stdev**2) of latent space
        outputs_logits, sampled_cm, mean, logvar = model(images)

        # calculate loss:
        seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, beta=beta, lossmode=lossmode)
        loss = seg_loss + kldloss

        # calculate the gradients:
        loss.backward()
        # update weights in model:
        optimizer.step()

        # calculate model prediction of reconstructed noisy label 
        train_output = calcPred(outputs_logits, sampled_cm)
        
        # calculate Dice
        train_dice = seg_score(labels_good.cpu().detach().numpy(), train_output.cpu().detach().numpy())

        # add to running score of metrics
        running_loss += seg_loss
        running_kld_loss += kldloss
        running_dice += train_dice

    # mean of metrics 
    av_loss = running_loss / num_batches
    av_kld = running_kld_loss / num_batches
    av_dice = running_dice / num_batches

    return av_loss, av_kld, av_dice

def train_lidc(model, device, train_loader, optimizer, epoch, num_epochs, ramp_up, beta, lossmode):
    """Training regime at one step of training for LIDC dataset

    Args:
        model (nn.Module): UNet_SCM our proposed model
        device (torch.device): device type responsible to load tensors into memory
        train_loader (torch.utils.data.DataLoader): iterable over MNIST training dataset
        optimizer (torch.optim.optimizer): PyTorch optimizer
        epoch (int): current epoch
        num_epochs (int): total number of epochs to train model 
        ramp_up (float): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value
        beta (int or float): KL-Divergence constant from beta-vae
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling

    Returns:
        av_loss (torch.Tensor): 1-element tensor containing average loss from training loop
        av_kld (torch.Tensor): 1-element tensor containing average KL-divergence from training loop 
        av_dice (torch.Tensor): 1-element tensor containing average DICE score from training loop
    """
    model.train()

    # initialize training metrics
    running_loss = 0
    running_kld_loss = 0
    running_dice = 0
    num_batches = len(train_loader)

    for j, (images, true_image, annots, imagename) in enumerate(train_loader):
        # zero graidents before each iteration
        optimizer.zero_grad()

        # cast numpy data into tensor float
        images = images.to(device=device, dtype=torch.float32)
        true_image = true_image.to(device=device, dtype=torch.float32)
        annots = annots.to(device=device, dtype=torch.float32)

        # model has four outputs:
        # 1: the probability map for true ground truth
        # 2: probability map for noisy ground truth
        # 3: mean of the latent space
        # 4: log variance (stdev**2) of latent space
        outputs_logits, sampled_cm, mean, logvar = model(images)
        
        # calculate loss:
        seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up, beta=beta, lossmode=lossmode)
        loss = seg_loss + kldloss
        
        # calclulate gradients:
        loss.backward()
        # update weights in model:
        optimizer.step()

        # calculate model prediction of reconstructed noisy label 
        train_output = calcPred(outputs_logits, sampled_cm)

        # calculate dice score of prediction
        train_dice = seg_score(true_image.cpu().detach().numpy(), train_output.cpu().detach().numpy())

        # add to running score of metrics
        running_loss += seg_loss
        running_kld_loss += kldloss
        running_dice += train_dice

    # mean of metrics
    av_loss = running_loss / num_batches
    av_kld = running_kld_loss / num_batches
    av_dice = running_dice / num_batches

    return av_loss, av_kld, av_dice


def calcPred(pred_seg_logits, cm):
    """Function to calculate our estimated ground truth with noise removed by confusion matrix

    Args:
        pred_seg_logits (torch.Tensor): probability map for ground truth 
        cm (torch.Tensor): sampled confusion matrix from variational autoencoder in model

    Returns:
        output (torch.Tensor): reconstructed ground truth, product between sampled cm and probability map
    """
    # get shape of probability map
    b, c, h, w = pred_seg_logits.size()

    # reshape probability map: [b, c, h, w] => [b*h*w, c, 1]
    pred_noisy = pred_seg_logits.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)

    # reshape cm: [b, h*w*c**2] => [ b∗h∗w, c , c ]
    anti_corrpution_cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
    # normalise confusion matrix along the rows:
    anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)
    # compute estimated annotators noisy seg proability with matrix multiplication
    outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b * h * w, c)
    # reshape: [b∗h∗w, c] => [b, c, h, w]
    outputs_clean = outputs_clean.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

    # get most likely class as binary mask
    _, output = torch.max(outputs_clean, dim=1)

    return output


def validate_stochastic_lidc(data_loader, model, device, epoch, num_epochs, ramp_up, beta, save_cm_path, save_img_path, lossmode):
    """Validation regime at one step of training for LIDC dataset

    Args:
        data_loader (torch.utils.data.DataLoader): iterable over LIDC validation dataset
        model (nn.Module): UNet_SCM our proposed model
        device (torch.device): device type responsible to load tensors into memory
        epoch (int): current epoch
        num_epochs (int): total number of epochs to train model 
        ramp_up (float, optional): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value.  Defaults to None.
        beta (int or float): KL-Divergence constant from beta-vae
        save_cm_path (_type_): folder path to save sampled confusion matrices over the course of model training
        save_img_path (str): folder path to save reconstruction ground truth samples over the course of model training
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling

    Returns:
        dice_met (2-tuple of float): 1- mean dice score and 2- standard deviation
        kld_met (2-tuple of float): 1- mean KL-divergence and 2- standard deviation
        seg_loss_met (2-tuple of float): 1- mean reconstruction loss and 2- standard deviation
        ged_met (2-tuple of float): 1- mean generalized energy distance and 2- standard deviation
    """
    model.eval()
    
    # collect metrics in list for mean and std calculation
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

            # model has four outputs:
            # 1: the probability map for true ground truth
            # 2: probability map for noisy ground truth
            # 3: mean of the latent space
            # 4: log variance (stdev**2) of latent space
            outputs_logits_original, sampled_cm, mean, logvar = model(images)

            # get shape of pred from segmentation network
            b, c, h ,w = outputs_logits_original.size()

            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up, beta=beta, lossmode=lossmode)

            # get annotator masks in a list
            noisy_labels = [annots[:,:,:,:,annot_idx].reshape(h, w).numpy() for annot_idx in range(annots.shape[4])]

            # get list of reconstructed segmentations with a different sampled cm for each one
            recon_segs = getSamples(outputs_logits_original, model, sample_no=len(noisy_labels))

            # calculate dice score with a reconstructed segmentation
            dice = seg_score(true_image.numpy(), recon_segs[0])

            # calculate generalized energy distance
            ged = generalized_energy_distance(noisy_labels, recon_segs)
            
            # add metrics to lsit
            ged_list.append(ged)
            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            
            # save progress of model every 10 epochs
            if epoch % 10 == 0:
                if j == 1:
                    # list of images to save
                    images_save = [images[:, 0, :, :].reshape(h, w).cpu().detach().numpy(), true_image.reshape(h, w).cpu().detach().numpy()] + recon_segs
                    save_name = save_img_path + '/test_' + str(epoch) + '.png'
                    #
                    plt.imsave(save_name, np.hstack(images_save), cmap='gray')

                # reshape cm: [b, h*w*c**2] => [b∗h∗w, c, c]
                cm = sampled_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
                
                save_path = save_cm_path + '/cm_' + str(epoch) + '.npy'
                np.save(save_path, cm.numpy())

    # mean and stdev of metrics
    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met

def validate_stochastic(data_loader, model, device, epoch, num_epochs, ramp_up, beta, save_cm_path, save_img_path, lossmode):
    """Validation regime at one step of training for MNIST dataset

    Args:
        data_loader (torch.utils.data.DataLoader): iterable over MNIST validation dataset
        model (nn.Module): UNet_SCM our proposed model
        device (torch.device): device type responsible to load tensors into memory
        epoch (int): current epoch
        num_epochs (int): total number of epochs to train model 
        ramp_up (float, optional): fraction of num_epochs where kl annealing gradually increases KL term to from 0 to its' actual value.  Defaults to None.
        beta (int or float): KL-Divergence constant from beta-vae
        save_cm_path (_type_): folder path to save sampled confusion matrices over the course of model training
        save_img_path (str): folder path to save reconstruction ground truth samples over the course of model training
        lossmode (str, optional): type of kl annealing to use, 'anneal' for standard annealing  or 'cyc' for cyclic kl annaeling

    Returns:
        dice_met (2-tuple of float): 1- mean dice score and 2- standard deviation
        kld_met (2-tuple of float): 1- mean KL-divergence and 2- standard deviation
        seg_loss_met (2-tuple of float): 1- mean reconstruction loss and 2- standard deviation
        ged_met (2-tuple of float): 1- mean generalized energy distance and 2- standard deviation
    """
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

            # collect all noisy labels
            labels_all = []
            labels_all.append(labels_over)
            labels_all.append(labels_under)
            labels_all.append(labels_wrong)
            labels_all.append(labels_good)

            # model has four outputs:
            # 1: the probability map for true ground truth
            # 2: probability map for noisy ground truth
            # 3: mean of the latent space
            # 4: log variance (stdev**2) of latent space
            outputs_logits_original, sampled_cm, mean, logvar = model(images)

            # shape of pred from segmentation network
            b, c, h ,w = outputs_logits_original.size()

            # calculate loss:
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits_original, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, beta=beta, lossmode=lossmode)
            
            # get all noisy annotator labels in list for ged calc
            noisy_labels = [labels_good.reshape(h,w).numpy(), labels_over.reshape(h, w).numpy(), labels_under.reshape(h, w).numpy(), labels_wrong.reshape(h, w).numpy()]

            # get list of reconstructed segmentations with a different sampled cm for each one
            recon_segs = getSamples(outputs_logits_original, model, sample_no=len(noisy_labels))

            # calculate dice score with reconstructed seg
            dice = seg_score(labels_good.numpy(), recon_segs[0])

            # calculate generalized energy distance
            ged = generalized_energy_distance(noisy_labels, recon_segs)

            seg_loss_list.append(seg_loss)
            kld_list.append(kldloss)
            dice_list.append(dice)
            ged_list.append(ged)

            # save progress of model every 10 epochs
            if epoch % 10 == 0:
                if j == 1:
                    # list of images to save
                    images_save = [images[:, 1, :, :].reshape(h, w), labels_good.reshape(h, w).cpu().detach().numpy()] + recon_segs
                    save_name = save_img_path + '/test_' + str(epoch) + '.png'
                    #
                    plt.imsave(save_name, np.hstack(images_save), cmap='gray')

                # reshape cm: [b, h*w*c**2] => [b∗h∗w, c, c]
                boo = sampled_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

                save_path = save_cm_path + '/cm_' + str(epoch) + '.npy'
                np.save(save_path, boo.numpy())

    # mean and stdev of metrics
    dice_met = eval_metric(dice_list)
    seg_loss_met = eval_metric(seg_loss_list)
    kld_met = eval_metric(kld_list)
    ged_met = eval_metric(ged_list)

    return dice_met, kld_met, seg_loss_met, ged_met


def test_stochastic(data_loader, model, device, save_path, sample_no=5):
    """Testing regime at one step of training for MNIST dataset

    Args:
        data_loader (torch.utils.data.DataLoader): iterable over MNIST testing dataset
        model (nn.Module): UNet_SCM our proposed model
        device (torch.device): device type responsible to load tensors into memory
        save_path (_type_): path to folder to save test images
        sample_no (int, optional): number of times to sample confusion matrices from model. Defaults to 5.

    Returns:
        dice_met (2-tuple of float): 1- mean dice score and 2- standard deviation
        ged_met (2-tuple of float): 1- mean generalized energy distance and 2- standard deviation
    """
    model.eval()

    dice_list = []
    ged_list = []
    with torch.no_grad():
        for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(data_loader):

            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            labels_over = labels_over.to(device=device, dtype=torch.float32)
            labels_under = labels_under.to(device=device, dtype=torch.float32)
            labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
            labels_good = labels_good.to(device=device, dtype=torch.float32)

            # collect all noisy labels
            labels_all = []
            labels_all.append(labels_over)
            labels_all.append(labels_under)
            labels_all.append(labels_wrong)
            labels_all.append(labels_good)

            # model outputs:
            # first one is the probability map for true ground truth
            outputs_logits_original, _, _, _ = model(images)

            # shape of pred from sgemtnation network
            b, c, h ,w = outputs_logits_original.size()

            # get list of reconstructed segmentations with a different sampled cm for each one
            recon_segs = getSamples(outputs_logits_original, model, sample_no=sample_no)

            # calculate dice score with reconstructed seg
            dice = seg_score(labels_good.numpy(), recon_segs[0])

            # get all noisy annotator labels in list for ged calc
            noisy_labels = [labels_good.reshape(h, w).numpy(), labels_over.reshape(h, w).numpy(), labels_under.reshape(h, w).numpy(), labels_wrong.reshape(h, w).numpy()]

            # arrays must be equal for ged calculation
            ged_noisy_segs = recon_segs[:len(noisy_labels)]

            # calculate generalized energy distance
            ged = generalized_energy_distance(noisy_labels, ged_noisy_segs)

            dice_list.append(dice)
            ged_list.append(ged)

            # list of images to save
            images_save = [images[:, 1, :, :].reshape(h, w), labels_good.reshape(h, w).cpu().detach().numpy()] + recon_segs
            save_name = save_path + '/test_' + str(j) + '.png'
            #
            plt.imsave(save_name, np.hstack(images_save), cmap='gray')

    # mean and stdev of metrics
    dice_met = eval_metric(dice_list)
    ged_met = eval_metric(ged_list)

    return dice_met, ged_met


def test_stochastic_lidc(data_loader, model, device, save_path, meta_df, sample_no=5):
    """testing regime our model adapted for the lidc dataset

    Args:
        data_loader (torch.utils.data.DataLoader): iterable over MNIST training dataset
        model (nn.Module): our proposed model
        device (torch.device): device type responsible to load tensors into memory
        save_path (str): folder path to save reconstructed segmentations
        meta_df (pandas.DataFrame): DataFrame containing metadata for LIDC-IDRI data
        sample_no (int, optional): number of times to sample confusion matrices from model. Defaults to 5.

    Returns:
        dice_met (2-tuple of float): 1- mean dice score and 2- standard deviation
        ged_met (2-tuple of float): 1- mean generalized energy distance and 2- standard deviation
    """
    model.eval()

    dice_list = []
    ged_list = []
    with torch.no_grad():
        for j, (images, true_image, annots, imagename) in enumerate(data_loader):
            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            annots = annots.to(device=device, dtype=torch.float32)

            # model outputs:
            # the probability map for true ground truth
            outputs_logits_original, _, _, _ = model(images)
            # 
            b, c, h ,w = outputs_logits_original.size()

            # get annotation labels as a list 
            noisy_labels = [annots[:,:,:,:,annot_idx].reshape(h, w).numpy() for annot_idx in range(annots.shape[4])]   

            # get reconstructions same length as noisy_labels for ged calc         
            recon_segs = getSamples(outputs_logits_original, model, sample_no=len(noisy_labels))

            # calculate dice score
            dice = seg_score(true_image.numpy(), recon_segs[0])
            
            # calculate generalized energy distance
            ged = generalized_energy_distance(noisy_labels, recon_segs)
            
            ged_list.append(ged)
            dice_list.append(dice)
            
            # get patient id and nodule number and subtlety for save name of images
            _,  patient_id, _, nod_no, _, _ = imagename[0].split('_')
            subtlety = meta_df.loc[(meta_df['patient_id'] == int(patient_id)) & (meta_df['nodule_no'] == int(nod_no))]['subtlety'].item()

            # list of images to save
            images_save = [images.reshape(h, w).cpu().detach().numpy(), true_image.reshape(h, w).cpu().detach().numpy()] + recon_segs
            # save to different subtlety folders
            save_name = save_path + str(subtlety) + '/test_' + str(j) + '.png'
            #
            plt.imsave(save_name, np.hstack(images_save), cmap='gray')

    # mean and std of metric
    dice_met = eval_metric(dice_list)
    ged_met = eval_metric(ged_list)

    return dice_met, ged_met


def getSamples(output_logits_original, model, sample_no=5):
    """Function to get list of reconstructed binary segmentations from sampled confusion matrices

    Args:
        output_logits_original (torch.Tensor): GT segmentation probability map, [b*h*w, c, 1]
        model (UNet_SCM): our proposed model
        sample_no (int, optional): number of times to sample confusion matrices from model. Defaults to 5.

    Returns:
        list of torch.Tensor: reconstructed segmentations from the product of sampled CMs and 'output_logits_original'
    """
    b, c, h, w = output_logits_original.size() # b: batch size, c: class number, h: height, w: width

    # sample CM from cm network of our model
    samples = model.cm_network.sample(sample_no)

    # iterate over sampled CMs and append reconstructed seg to list
    sample_list = []
    for sample in samples:
        # add batch size, b, dimension
        sample_cm = torch.unsqueeze(sample, dim=0)

        # calcuate reconstruction
        output = calcPred(output_logits_original, sample_cm)

        sample_list.append(output.reshape(h, w).cpu().detach().numpy())

    return sample_list

