import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import LowRankMultivariateNormal, Normal

from Utilis import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet_SCM(nn.Module):
    """ Proposed method containing a segmentation network and a confusion matrix network.
    The segmentation network is U-net. The "annotation network" is defined in cm_network

    Args:
        in_ch (int): dimension of input
        resolution (int): resolution of input
        width (int): number of channels in the first encoder for U-Net architecture portion
        depth (int): numver of down-sampling stages - 1
        class_no (int): number of output classes, 2 for all use in our work
        latent (int, optional): dimension of latent space for VAE. Defaults to 6
        norm (str, optional): normalization to use. Defaults to 'in' or instance normalization
    """
    def __init__(self, in_ch, resolution, width, depth, class_no, latent=6, norm='in'):
        """Constructor of UNet_SCM class.
        """ 
        super(UNet_SCM, self).__init__()
        self.depth = depth
        self.final_in = class_no

        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        self.cm_network = cm_net(c=width, h=resolution, w=resolution, class_no=class_no, latent=latent)

        # build encoders and decoders according to U-Net architecture
        for i in range(self.depth):

            if i == 0:

                self.encoders.append(double_conv(in_channels=in_ch, out_channels=width, step=1, norm=norm))
                self.decoders.append(double_conv(in_channels=width*2, out_channels=width, step=1, norm=norm))

            elif i < (self.depth - 1):

                self.encoders.append(double_conv(in_channels=width*(2**(i - 1)), out_channels=width*(2**i), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**(i + 1)), out_channels=width*(2**(i - 1)), step=1, norm=norm))

            else:

                self.encoders.append(double_conv(in_channels=width*(2**(i-1)), out_channels=width*(2**(i-1)), step=2, norm=norm))
                self.decoders.append(double_conv(in_channels=width*(2**i), out_channels=width*(2**(i - 1)), step=1, norm=norm))

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_last = nn.Conv2d(width, self.final_in, 1, bias=True)

    def forward(self, x):
        """Forward process of our model

        Args:
            x (torch.Tensor): input images

        Returns:
            y_t (torch.Tensor): segmentation network output
            sampled_cm (torch.Tensor): sampled confusion matrix from cm_network 
            mu (torch.Tensor): mean of encoded distribution
            log_var (torch.Tensor): output log variance of encoder distribution
        """

        y = x
        encoder_features = []

        # train according to U-Net architecture
        for i in range(len(self.encoders)):

            y = self.encoders[i](y)
            encoder_features.append(y)

        for i in range(len(encoder_features)):

            y = self.upsample(y)
            y_e = encoder_features[-(i+1)]

            if y_e.shape[2] != y.shape[2]:
                diffY = torch.tensor([y_e.size()[2] - y.size()[2]])
                diffX = torch.tensor([y_e.size()[3] - y.size()[3]])

                y = F.pad(y, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

            y = torch.cat([y_e, y], dim=1)
            y = self.decoders[-(i+1)](y)

        # final convolution layer for segmentation network
        y_t = self.conv_last(y)

        # send output to VAE structure for generating a confusion matrix
        sampled_cm, mu, log_var = self.cm_network(y)

        return y_t, sampled_cm, mu, log_var



class cm_net(nn.Module):
    """ This class defines the stochastic annotator network which is a variational auto encoder

    Args: 
        c (int): channels of input
        h (int): height of input
        w (int): width of input
        class_no (int, optional): number of classes to segment. Defaults to 2 because this is binary segmentation
        latent (int, optional): Dimension size of latent space. Defaults to 6
    """
    def __init__(self, c, h, w, class_no=2, latent=6):
        """Constructor of cm_network class
        """
        super(cm_net, self).__init__()
        self.fc_encoder = nn.Linear(c * h * w, latent // 2)
        self.fc_decoder = nn.Linear(latent, h * w * class_no ** 2)
        self.fc_mu = nn.Linear(latent // 2, latent)
        self.fc_var = nn.Linear(latent // 2, latent)

        self.latent = latent

    def forward(self, x):
        """Forward process of our cm network

        Args:
            x (torch.Tensor): input img

        Returns:
            cm (torch.Tensor): reconstructed confusion matrix
            mu (torch.Tensor): mean of distribution
            log_var (torch.Tensor): log variance of distribution
        """

        # encode into latent space adn get parameters of latent distribution
        mu, log_var = self.encode(x)

        # reparameterization trick to allow backprop
        z = self.reparameterize(mu, log_var)

        # decode to reconstruct confusion matrix
        cm = self.decode(z)
        
        return cm, mu, log_var

    def encode(self, input):
        """Encodes input into latent space

        Args:
            input ():

        Returns:
            mu ():
            log_var ():
        """
        res = torch.flatten(input, start_dim=1)
        res = F.relu(self.fc_encoder(res))

        # mu and var components of the latent distribution
        mu = self.fc_mu(res)
        log_var = self.fc_var(res)

        return mu, log_var

    def decode(self, z):
        """Maps latent space sample onto the image space

        Args:
            z (torch.Tensor): sampled latent vector

        Returns:
            res (torch.Tensor): final confusion matrix
        """
        res = self.fc_decoder(z)
        res = F.softplus(res)
        return res

    def reparameterize(self, mu, log_var):
        """Reparameterization trick to sample from N(mu, var) from
        N(0,1).

        Args:
            mu (torch.Tensor): mean of distribution
            log_var (torch.Tensor): log variance of distribution

        returns:
            z (torch.Tensor): sampled latent vector
        """
        std = torch.exp(0.5*log_var) # recover std from log of var
        eps = torch.randn_like(std) # random noise

        # reparameterization formula
        z = eps*std + mu 

        return z

    def sample(self, num_samples):
        """
        Samples from the latent space and return the corresponding
        image space map.

        Args:
            num_samples (int): number of times to sample from latent space
        """
        z = torch.randn(num_samples, self.latent)
        z = z.to(device)
        samples = self.decode(z)
        return samples

def double_conv(in_channels, out_channels, step, norm):
    # ===========================================
    # in_channels: dimension of input
    # out_channels: dimension of output
    # step: stride
    # ===========================================
    if norm == 'in':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'ln':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels, out_channels, affine=True),
            nn.PReLU()
        )
    elif norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=step, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, groups=1, bias=False),
            nn.GroupNorm(out_channels // 8, out_channels, affine=True),
            nn.PReLU()
        )