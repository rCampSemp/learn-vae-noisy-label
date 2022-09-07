import os
import errno

import torch
import numpy as np
from adamW import AdamW
from torch.utils import data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from Stochastic_Loss import stochastic_noisy_label_loss
from Utilis import seg_score, CustomDataset_punet
from torch.nn import functional as F

# our proposed model:
from Stochastic_CM import UNet_SCM


if __name__ == '__main__':
    # ========================= #
    # Hyper-parameters setting
    # ========================= #

    # hyper-parameters for model:
    input_dim = 3  # dimension of input
    width = 24  # width of the network
    depth = 3  # depth of the network, downsampling times is (depth-1)
    class_no = 2  # class number, 2 for binary

    # hyper-parameters for training:
    train_batchsize = 5  # batch size
    alpha = 1.0  # weight of the kl loss
    num_epochs = 100  # total epochs
    latent = 512
    learning_rate = 1e-3  # learning rate DO NOT USE 1E-2!!
    ramp_up = 0.3 # This ramp up is necessary!!!
    num_samples = 5 # Number of CM samples to take in testing

    # image resolution:
    mnist_resolution = 28

    # ======================================= #
    # Prepare a few data examples from MNIST
    # ======================================= #

    # Change path for your own datasets here:
    data_path = './MNIST_examples'
    dataset_tag = 'mnist'
    label_mode = 'multi'

    # full path to train/validate/test:
    test_path = data_path + '/test'
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'

    # prepare data sets using our customdataset
    train_dataset = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=True)
    validate_dataset = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)
    test_dataset = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)

    # putting dataset into data loaders
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # demonstrate the training samples:
    Image_index_to_demonstrate = 6
    images, labels_over, labels_under, labels_wrong, labels_good, imagename = validate_dataset[Image_index_to_demonstrate]
    images = np.mean(images, axis=0)

    # call model:
    model = UNet_SCM(in_ch=input_dim,
                     resolution=mnist_resolution,
                     width=width,
                     depth=depth,
                     latent=latent,
                     class_no=class_no,
                     norm='in')

    # model name for saving:
    model_name = 'UNet_NoUnder_Conditional_Stochastic_Confusion_Matrices_' + '_width' + str(width) + \
                 '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                 '_alpha_' + str(alpha) + '_e' + str(num_epochs) + \
                 '_lr' + str(learning_rate)

    # setting up device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # =================================================== #
    # Prepare folders to save trained models and results
    # =================================================== #

    # save location:
    saved_information_path = './Results/MNIST/Ablation'
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_information_path = saved_information_path + '/' + model_name
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    save_path_visual_result = saved_information_path + '/visual_results'
    try:
        os.mkdir(save_path_visual_result)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    # tensorboardX file saved location:
    writer = SummaryWriter('./Results/MNIST/Ablation/Log/Log_' + model_name)

    # =================================================== #
    # Training
    # =================================================== #

    # We use adamW optimiser for more accurate L2 regularisation
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0
        running_kld_loss = 0
        running_iou = 0

        outputs = 0
        for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(trainloader):
            b, c, h, w = images.size()

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
            seg_loss, kldloss = stochastic_noisy_label_loss(outputs_logits, sampled_cm, mean, logvar, labels_all, epoch, num_epochs, data='mnist', ramp_up=ramp_up, beta=alpha)
            # print(images.size())
            loss = seg_loss + kldloss
            # calculate the gradients:
            # loss = lossss(model, sampled_outputs_logits, stochastic_cm, mean, logvar, label, epoch, num_epochs, ramp_up, alpha)
            loss.backward()
            # update weights in model:
            optimizer.step()

            # Now outputs_logits is the noisy seg:
            b_, c_, h_, w_ = outputs_logits.size()
            # pred_norm_prob_noisy = nn.Softmax(dim=1)(outputs_logits)
            pred_noisy = outputs_logits.view(b_, c_, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_, 1)
            anti_corrpution_cm = sampled_cm.view(b_, c_ ** 2, h_ * w_).permute(0, 2, 1).contiguous().view(b_ * h_ * w_, c_ * c_).view(b_ * h_ * w_, c_, c_)
            anti_corrpution_cm = torch.softmax(anti_corrpution_cm, dim=1)
            outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b_ * h_ * w_, c_)
            outputs_clean = outputs_clean.view(b_, h_ * w_, c_).permute(0, 2, 1).contiguous().view(b_, c_, h_, w_)

            _, train_output = torch.max(outputs_clean, dim=1)

            train_iou = seg_score(labels_good.cpu().detach().numpy(), train_output.cpu().detach().numpy())
            running_loss += seg_loss
            running_kld_loss += kldloss
            running_iou += train_iou

            if (j + 1) == 1:
                # check the validation accuray at the begning of each epoch:
                # v_dice = evaluate(data=validateloader,
                #                   model=model,
                #                   class_no=class_no)

                print(
                    'Step [{}/{}], '
                    'Train dice: {:.4f},'
                    'loss main: {:.4f},'
                    'loss kl: {:.4f},'.format(epoch + 1, num_epochs,
                                              train_iou,
                                              running_loss / (j + 1),
                                              running_kld_loss / (j + 1)))

                writer.add_scalars('scalars', {'train main loss': running_loss / (j + 1),
                                               'train kld loss': running_kld_loss / (j + 1),
                                               'train iou': running_iou / (j + 1)}, epoch + 1)

    # save model:
    save_model_name_full = saved_model_path + '/' + model_name + '_Final.pt'
    torch.save(model, save_model_name_full)
    print('\n')
    print('Training ended')
    
    model.eval()
    running_test_dice = 0
    with torch.no_grad():
        for i, (v_images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(testloader):
            v_images = v_images.to(device=device, dtype=torch.float32)
            v_outputs_logits_original, sampled_cm, _, __ = model(v_images)
            b, c, h, w = v_outputs_logits_original.size()

            # plot the final segmentation map
            samples = model.cm_network.sample(num_samples)
            
            # print(v_stochastic_cm.size())
            # pred_norm_prob_noisy = nn.Softmax(dim=1)(v_outputs_logits_original)
            pred_noisy = v_outputs_logits_original.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
            
            sample_list = [v_images[:, 1, :, :].reshape(h, w).cpu().detach().numpy(), labels_good.reshape(h, w).cpu().detach().numpy()]
            av_sampled_dice = 0
            for k, sample in enumerate(samples):
                sample_cm = torch.unsqueeze(sample, dim=0)

                anti_corrpution_cm = sample_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
                # anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)
                anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)

                # pred_norm_prob_noisy = pred_norm_prob_noisy.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
                outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b * h * w, c)
                v_outputs_logits_original = outputs_clean.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

                # v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
                _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)

                test_dice = seg_score(labels_good, v_outputs_logits)
                av_sampled_dice += test_dice / num_samples

                # save_name = save_path_visual_result + '/test_' + str(i) + '_seg_' + str(k) + '.png'
                sample_list.append(v_outputs_logits.reshape(h, w).cpu().detach().numpy())
                # plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            running_test_dice += av_sampled_dice / len(testloader)

            save_name_samples = save_path_visual_result + '/test_' + str(i)  + '.png'
            plt.imsave(save_name_samples, np.hstack(sample_list), cmap='gray')

            # save_name_slice = save_path_visual_result + '/test_' + str(i) + '_img.png'
            # plt.imsave(save_name_slice, v_images[:, 1, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')

            # save_name_label = save_path_visual_result + '/test_' + str(i) + '_label.png'
            # plt.imsave(save_name_label, labels_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')

    results = 'Test Dice:' + str(running_test_dice)
    ff_path = saved_information_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(results)
    ff.close()
