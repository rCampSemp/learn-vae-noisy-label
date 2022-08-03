import enum
import os
import errno
import torch
import timeit
import imageio
import numpy as np
import torch.nn as nn
from adamW import AdamW
import torch.optim as optim
from torch.utils import data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from Deterministic_Loss import deterministic_noisy_label_loss
from Utilis import segmentation_scores, CustomDataset_LIDC, calculate_cm
from Utilis import LIDC_collate
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd

# our proposed model:
from Deterministic_CM import UNet_DCM
from PIL import Image

if __name__ == '__main__':
    # ========================= #
    # Hyper-parameters setting
    # ========================= #

    # hyper-parameters for model:
    input_dim = 1  # dimension of input
    width = 8  # width of the network 
    depth = 3  # depth of the network, downsampling times is (depth-1)
    class_no = 2  # class number, 2 for binary

    # hyper-parameters for training:
    train_batchsize = 5  # batch size 
    num_epochs = 15  # total epochs
    learning_rate = 1e-3  # learning rate DO NOT USE 1E-2!!
    ramp_up = 0.2 # This ramp up is necessary!!!

    # ======================================= #
    # Prepare a few data examples from LIDC 
    # ======================================= #

    # Change path for your own datasets here:
    data_path = './LIDC_examples'

    # get meta file as dataframe
    meta_file = data_path + '/meta/metadata.csv'
    meta_df = pd.read_csv(meta_file)

    # full path to train/validate/test:
    test_path = data_path + '/test'
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'

    # prepare data sets using our customdataset
    train_dataset = CustomDataset_LIDC(dataset_location=train_path, augmentation=True)
    validate_dataset = CustomDataset_LIDC(dataset_location=validate_path, augmentation=False)
    test_dataset = CustomDataset_LIDC(dataset_location=test_path, augmentation=False)

    # putting dataset into data loaders
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, collate_fn=LIDC_collate, drop_last=True)
    validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, drop_last=False)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    # call model:
    model = UNet_DCM(in_ch=input_dim,
                     resolution=64,
                     width=width,
                     depth=depth,
                     latent=512,
                     class_no=class_no,
                     norm='in')

    # model name for saving:
    model_name = 'LIDC_UNet_Conditional_Deterministic_Confusion_Matrices_' + '_width' + str(width) + \
                 '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                 '_epoch' + str(num_epochs) + \
                 '_ramp' + str(ramp_up) + \
                 '_lr' + str(learning_rate)

    # # setting up device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # =================================================== #
    # Prepare folders to save trained models and results
    # =================================================== #

    # save location:
    saved_information_path = './Results'
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
    save_path_subtlety = save_path_visual_result + '/subtlety_'
    try:
        os.mkdir(save_path_visual_result)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    for i in range(1, 6):
        try:
            os.mkdir(save_path_subtlety + str(i))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


    # tensorboardX file saved location:
    writer = SummaryWriter('./Results/Log_' + model_name)

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
        for j, (images, true_image, annots, imagename) in enumerate(trainloader):
            b, c, h, w = images.size()

            # zero graidents before each iteration
            optimizer.zero_grad()

            # cast numpy data into tensor float
            images = images.to(device=device, dtype=torch.float32)
            true_image = true_image.to(device=device, dtype=torch.float32)
            annots = annots.to(device=device, dtype=torch.float32)

            # model has two outputs:
            # first one is the probability map for true ground truth
            # second one is a list collection of probability maps for different noisy ground truths
            outputs_logits, stochastic_cm = model(images)
            # outputs = 0.9*outputs + 0.1*outputs_logits

            # calculate loss:
            loss = deterministic_noisy_label_loss(outputs_logits, stochastic_cm, annots, epoch, num_epochs, data='lidc', ramp_up=ramp_up)
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
            train_iou = segmentation_scores(true_image.cpu().detach().numpy(), train_output.cpu().detach().numpy(), class_no)
            running_loss += loss
            running_iou += train_iou
            if (j + 1) == 1:
                # check the validation accuray at the begning of each epoch:
                # v_dice = evaluate(data=validateloader,
                #                   model=model,
                #                   class_no=class_no)

                print(
                    'Step [{}/{}], '
                    'Train dice: {:.4f},'
                    'loss main: {:.4f},'.format(epoch + 1, num_epochs,
                                              train_iou,
                                              running_loss / (j + 1)))

                writer.add_scalars('scalars', {'train main loss': running_loss / (j + 1),
                                               'train iou': running_iou / (j + 1)}, epoch + 1)

    # save model:
    save_model_name_full = saved_model_path + '/' + model_name + '_Final.pt'
    torch.save(model, save_model_name_full)
    print('\n')
    print('Training ended')

    model.eval()
    for i, (v_images, true_image, annots, imagename) in enumerate(testloader):
        v_images = v_images.to(device=device, dtype=torch.float32)
        v_outputs_logits_original, v_stochastic_cm = model(v_images)
        b, c, h, w = v_outputs_logits_original.size()
        # plot the final segmentation map

        # print(v_stochastic_cm.size())
        # pred_norm_prob_noisy = nn.Softmax(dim=1)(v_outputs_logits_original)
        pred_noisy = v_outputs_logits_original.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
        anti_corrpution_cm = v_stochastic_cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        # anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)
        anti_corrpution_cm = anti_corrpution_cm / anti_corrpution_cm.sum(1, keepdim=True)

        # pred_norm_prob_noisy = pred_norm_prob_noisy.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
        outputs_clean = torch.bmm(anti_corrpution_cm, pred_noisy).view(b * h * w, c)
        v_outputs_logits_original = outputs_clean.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        # v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)
        
        _, patient_id, _, nod_no, _, _ = imagename[0].split('_')
        subtlety = meta_df.loc[(meta_df['patient_id'] == int(patient_id)) & (meta_df['nodule_no'] == int(nod_no))]['subtlety'].item()
        
        save_name = save_path_subtlety + str(subtlety) + '/test_' + str(i) + '_seg.png'
        save_name_label = save_path_subtlety + str(subtlety) + '/test_' + str(i) + '_label.png'
        save_name_slice = save_path_subtlety + str(subtlety) + '/test_' + str(i) + '_img.png'

        plt.imsave(save_name_slice, v_images.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        plt.imsave(save_name_label, true_image.reshape(h, w).cpu().detach().numpy(), cmap='gray')

        # # plot the noisy segmentation maps:
        # v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h * w)
        # v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
        # v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b * h * w, c, 1)
        # for j, cm in enumerate(v_outputs_logits_noisy):
        #     cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        #     cm = cm / cm.sum(1, keepdim=True)
        #     v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b * h * w, c)
        #     v_noisy_output_original = v_noisy_output_original.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        #     _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
        #     save_name = save_path_visual_result + '/test_' + str(i) + '_noisy_' + str(j) + '_seg.png'
        #     plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')

    subtlety = 2
    test_data_index = 0

    seg = save_path_subtlety + str(subtlety) + '/test_' + str(test_data_index) + '_seg.png'
    label = save_path_subtlety + str(subtlety) + '/test_' + str(test_data_index) + '_label.png'
    img = save_path_subtlety + str(subtlety) + '/test_' + str(test_data_index) + '_img.png'
    
    # plot image, ground truth and final segmentation
    fig = plt.figure(figsize=(6.7, 13))
    columns = 3
    rows = 1

    ax = []
    imgs = []
    imgs_names = []

    imgs.append(img)
    imgs.append(label)
    imgs.append(seg)

    imgs_names.append('Test img')
    imgs_names.append('GroundTruth')
    imgs_names.append('Pred of true seg')

    for i in range(columns * rows):
        img_ = imgs[i]
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(imgs_names[i])
        img_ = Image.open(img_)
        img_ = np.array(img_, dtype='uint8')
        plt.imshow(img_, cmap='gray')
    plt.show()
