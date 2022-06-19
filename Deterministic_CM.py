import torch
import torch.nn as nn
import torch.nn.functional as F

from Utilis import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet_DCM(nn.Module):
    """ Proposed method containing a segmentation network and a confusion matrix network.
    The segmentation network is U-net. The confusion  matrix network is defined in cm_layers

    """
    def __init__(self, in_ch, resolution, batch_size, input_dim, width, depth, class_no, latent=512, norm='in'):
        #
        # ===============================================================================
        # in_ch: dimension of input
        # class_no: number of output class
        # width: number of channels in the first encoder
        # depth: down-sampling stages - 1
        # ===============================================================================
        super(UNet_DCM, self).__init__()
        self.depth = depth
        self.noisy_labels_no = 4
        self.final_in = class_no

        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        self.scm_encoder = scm_encoder(c=width, h=resolution, w=resolution, latent=latent)
        self.scm_decoder = scm_decoder(c=input_dim, h=resolution, w=resolution, class_no=class_no, latent=latent)

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

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, num_samples):
        # num_samples will be used as the same shape as the z in forward function
        z = torch.randn(num_samples)
        z = z.to(device)
        cms = self.scm_decoder(z)
        return cms

    def forward(self, x):

        y = x
        encoder_features = []

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

        y_t = self.conv_last(y)
        # mu, logvar = self.scm_encoder(y)
        cm = self.scm_encoder(y)
        # z = self.reparameterize(mu, logvar)
        # cm = self.scm_decoder(z)
        cm = self.scm_decoder(cm)

        return y_t, cm, 0, 0


class scm_encoder(nn.Module):
    """ This class defines the stochastic annotator network
    """
    def __init__(self, c, h, w, latent):
        super(scm_encoder, self).__init__()
        # self.fc_mu = nn.Linear(c*h*w, latent)
        # self.fc_var = nn.Linear(c*h*w, latent)
        self.fc = nn.Linear(c * h * w, latent)

    def forward(self, x):
        # print(x.size())
        y = torch.flatten(x, start_dim=1)
        y = self.fc(y)
        # mu = self.fc_mu(y)
        # var = self.fc_var(y)

        # return mu, var
        return y


class scm_decoder(nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """
    def __init__(self, c, h, w, class_no=2, latent=512):
        super(scm_decoder, self).__init__()
        self.w = w
        self.h = h
        self.class_no = class_no
        self.mlp_cm = nn.Linear(latent, h*w*class_no**2) # pixel wise
        # self.mlp_cm = nn.Linear(latent, class_no ** 2) # global

    def forward(self, x):
        cm = self.mlp_cm(x)
        cm = F.softplus(cm)

        return cm


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