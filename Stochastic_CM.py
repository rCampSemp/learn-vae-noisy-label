import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import LowRankMultivariateNormal

from Utilis import init_weights, init_weights_orthogonal_normal, l2_regularisation
from torch.distributions import Normal, Independent, kl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UNet_SCM(nn.Module):
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
        super(UNet_SCM, self).__init__()
        self.depth = depth
        self.noisy_labels_no = 4
        self.final_in = class_no

        self.decoders = nn.ModuleList()
        self.encoders = nn.ModuleList()

        # print(width*(2**(depth-1)))
        # print(resolution // (2**(depth-1)))

        # self.vae_encoder = scm_encoder(c=self.final_in, h=resolution, w=resolution, latent=latent)
        self.vae_encoder = scm_encoder(c=self.final_in, h=resolution, w=resolution, latent=latent)
        self.vae_decoder = scm_decoder(c=self.final_in, h=resolution, w=resolution, latent=latent)

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

        self.conv_cm = nn.Conv2d(self.final_in, class_no**2, 1, bias=True)
        # self.conv_last = seg_head(c=width, h=resolution, w=resolution, class_no=class_no, latent=latent)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss
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

    def forward(self, x, gt):

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

        seg = self.conv_last(y)
        # conditional_seg = torch.cat(seg, y)
        if self.vae_encoder.training is True:
            gt = gt.repeat(1, self.final_in, 1, 1)
            mu, logvar = self.vae_encoder(gt)
        else:
            mu, logvar = self.vae_encoder(seg)
        z = self.reparameterize(mu, logvar)
        seg_sample = self.vae_decoder(z)

        # seg = self.conv_last(y)
        cm = self.conv_cm(seg)

        return seg, seg_sample, cm, mu, logvar


class scm_encoder(nn.Module):
    """ This class defines the stochastic annotator network
    """
    def __init__(self, c, h, w, latent):
        super(scm_encoder, self).__init__()
        self.fc_mu = nn.Linear(c*h*w, latent)
        self.fc_var = nn.Linear(c*h*w, latent)

    def forward(self, x):
        y = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(y)
        var = self.fc_var(y)
        return mu, var


class scm_decoder(nn.Module):
    """ This class defines the annotator network, which models the confusion matrix.
    Essentially, it share the semantic features with the segmentation network, but the output of annotator network
    has the size (b, c**2, h, w)
    """
    def __init__(self, c, h, w, latent=512):
        super(scm_decoder, self).__init__()
        self.w = w
        self.h = h
        self.c = c
        self.mlp_seg = nn.Linear(latent, h*w*c)

    def forward(self, x):
        y = self.mlp_seg(x)
        return y.reshape(-1, self.c, self.h, self.w)

#
# class seg_head(nn.Module):
#     def __init__(self, c, h, w, class_no, latent):
#         super(seg_head, self).__init__()
#         self.encoder = nn.Linear(c*h*w, latent)
#         self.decoder = nn.Linear(latent, h*w*class_no**2)
#
#     def forward(self, x):
#         y = torch.flatten(x, start_dim=1)
#         y = self.encoder(y)
#         y = F.relu(y, inplace=True)
#         y = self.decoder(y)
#         return y


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