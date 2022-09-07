import torch
from Train_punet import train_punet
from Train_VAE import trainStoch
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':
    # ========================================================
    # comment out other train functions when u use one of them
    # ============================================
    # for probabilistic u-net
    # ============================================
    # train_punet(epochs=5,
    #             repeat=1,
    #             train_batch_size=5,
    #             lr=1e-4,
    #             num_filters=[4, 8, 16, 32],
    #             input_channels=1,
    #             latent_dim=6,
    #             no_conv_fcomb=2,
    #             num_classes=2,
    #             beta=5,
    #             test_samples_no=10,
    #             dataset_path='./LIDC_examples',
    #             dataset_tag='lidc')
    # ==========================================
    # for training with our model with VAE CM with MNIST
    # ==========================================
    # trainStoch(input_dim=3,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=5,
    #             validate_batchsize=1,
    #             num_epochs=1,
    #             learning_rate=1e-3,
    #             width=24,
    #             depth=3,
    #             resolution=28,
    #             data_path='./MNIST_examples',
    #             dataset_tag='mnist',
    #             label_mode='multi',
    #             ramp_up=0.3,
    #             beta=20.0,
    #             lossmode='anneal')
    # ==========================================
    # for training with our model with VAE CM with LIDC
    # ==========================================
    trainStoch(input_dim=1,
                class_no=2,
                repeat=1,
                train_batchsize=5,
                validate_batchsize=1,
                num_epochs=1,
                learning_rate=1e-4,
                width=24,
                depth=3,
                resolution=64,
                data_path='./LIDC_examples',
                dataset_tag='lidc',
                ramp_up=0.3,
                beta=20.0,
                lossmode='anneal')
