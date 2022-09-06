import torch
import os
import errno
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from load_LIDC_data import LIDC_IDRI
from Models import ProbabilisticUnet
import pandas as pd

from Utilis import CustomDataset_punet, LIDC_collate, test_punet, evaluate_punet, l2_regularisation, CustomDataset_LIDC
from adamW import AdamW
# ===================
# main computation:
# ===================
def getData(train_batchsize, data_path, dataset_tag):
    #
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'
    #
    if dataset_tag == 'lidc':
        # 
        train_dataset = CustomDataset_LIDC(dataset_location=train_path , model='punet', augmentation=True)
        validate_dataset = CustomDataset_LIDC(dataset_location=validate_path, model='ours', augmentation=False)
        test_dataset = CustomDataset_LIDC(dataset_location=test_path, model='ours', augmentation=False)
        # 
        # Dataloaders with custom collate_fn for LIDC data
        trainloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=4, drop_last=True)
        validateloader = DataLoader(validate_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=LIDC_collate)
        testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, collate_fn=LIDC_collate)
        #
        return trainloader, validateloader, testloader, train_dataset
    #
    elif dataset_tag == 'mnist':
        dataset_train = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel='p_unet', augmentation=True)
        dataset_val = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel='multi', augmentation=False)
        dataset_test = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel='multi', augmentation=False)

        train_loader = DataLoader(dataset_train, batch_size=train_batchsize, shuffle=True, num_workers=4, drop_last=True)
        val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
        #
        return train_loader, val_loader, test_loader, dataset_train


def train_punet(epochs,
                repeat,
                train_batch_size,
                lr,
                num_filters,
                input_channels,
                latent_dim,
                no_conv_fcomb,
                num_classes,
                beta,
                test_samples_no,
                dataset_path,
                dataset_tag):
    """Runs full trai(ning script for probabilistic unet model.

    Args:
        epochs (int): training epoch length
        repeat (int): number of times to run script
        train_batch_size (int): batch size used during training
        lr (float): learning rate 
        num_filters (list): dimensions of architecture
        input_channels (int): channel number of input image
        latent_dim (int): dimensions of latent space
        no_conv_fcomb (int): number of 1x1 convolutions in defined in fcomb class in punet model
        num_classes (int): number of classes in segmentation task. 2 for our work (binary segmentation)
        beta (int): KL-Divergence constant from beta-vae
        test_samples_no (int): number of times to sample a reconstructed segmentation
        dataset_path (str): folder path to parent folder of data used
        dataset_tag (str): dataset to use. 'mnist' for MNIST, and 'lidc' for LIDC-IDRI data
    """
    for itr in range(repeat):
        # # dictionary for results graphs
        # vals = {'Epoch': [], 'Dice': [], 'GED': [], 'Loss': [], 'KL-Div': []}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get dataloaders for training, validation and testing 
        train_loader, val_loader, test_loader, dataset_train = getData(train_batch_size, dataset_path, dataset_tag) 

        # probabilistic unet model
        net = ProbabilisticUnet(input_channels=input_channels, num_classes=num_classes, num_filters=num_filters, latent_dim=latent_dim, no_convs_fcomb=no_conv_fcomb, beta=beta)
        net.to(device)

        optimizer = AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

        training_iterations = len(dataset_train) // train_batch_size - 1

        for epoch in range(epochs):
            #
            net.train()
            #
            for step, (patch, mask, name) in enumerate(train_loader):
                # store in device
                patch = patch.to(device)
                mask = mask.to(device)

                # train model
                net.forward(patch, mask, training=True)
                # get loss terms
                elbo, reconstruction, kl = net.elbo(mask)
                # regularisation term
                reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

                # update weights and gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #
                if (step + 1) == training_iterations:
                    # validation loop
                    validate_dice, generalized_energy_distance_epoch = evaluate_punet(net=net, val_data=val_loader, class_no=num_classes, sampling_no=3, datatag=dataset_tag)
                    print('epoch:' + str(epoch))
                    # vals['Epoch'].append(epoch)
                    #
                    print('val dice: ' + str(validate_dice))
                    # vals['Dice'].append(validate_dice[0])
                    #
                    print('val generalized_energy: ' + str(generalized_energy_distance_epoch))
                    # vals['GED'].append(generalized_energy_distance_epoch[0])
                    #
                    print('train loss: ' + str(loss.item()))
                    # vals['Loss'].append(loss.item())
                    #
                    print('kl is: ' + str(kl.item()))
                    # vals['KL-Div'].append(kl.item())
                    #
                    print('reconstruction loss is: ' + str(reconstruction.item()))
                    print('\n')
        print('\n')

        # create save path folder
        save_path = '../Results_PUnet'
        # 
        try:
            #
            os.mkdir(save_path)
            #
        except OSError as exc:
            #
            if exc.errno != errno.EEXIST:
                #
                raise
            #
            pass
        #
        save_path = save_path + '/150' + str(itr) + \
                    '_punet_' + \
                    '_train_batch_' + str(train_batch_size) + \
                    '_latent_dim_' + str(latent_dim) + \
                    '_lr_' + str(lr) + \
                    '_epochs_' + str(epochs) + \
                    '_beta_' + str(beta) + \
                    '_test_sample_no_' + str(test_samples_no)
        #
        test_punet(net=net, testdata=test_loader, save_path=save_path, sampling_times=test_samples_no, datatag=dataset_tag)
        #
        # ## plot graphs
        # save_res_path = save_path + '/full_plot.png'

        # df = pd.DataFrame.from_dict(vals)
        # plt.figure()
        # ax = plt.gca()
        # df.plot(kind='line', x='Epoch', y='Dice', ax=ax)
        # df.plot(kind='line', x='Epoch', y='GED', color='red', ax=ax)
        # df.plot(kind='line', x='Epoch', y='Loss', color='cyan', ax=ax)
        # ax.get_figure().savefig(save_res_path)
        # #
        # save_kld_path = save_path + '/kld_plot.png'
        # plt.figure()
        # ax2 = plt.gca()
        # df.plot(kind='line', x='Epoch', y='KL-Div', ax=ax2)
        # ax2.get_figure().savefig(save_kld_path)
        #
    print('Training is finished.')
#




