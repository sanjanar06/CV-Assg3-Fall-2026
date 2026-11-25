# Implementation of flow matching described in https://arxiv.org/abs/2209.03003

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import build_arch
from .guidance import sampling_mode, validate_guidance
from .tiny_autoencoder import TAESD


class FM(nn.Module):
    """Flow Matching Model"""

    def __init__(
        self,
        img_shape=(3, 32, 32),
        timesteps=100,
        num_classes=10,
        use_vae=False,
        vae_encoder_weights=None,
        vae_decoder_weights=None,
        arch_type="UNet",
        arch_cfg=None,
    ):
        """
        Args:
            img_shape (tuple/list of int): image shape before any VAE (C x H x W)
            timesteps (int): number of timesteps in flow matching
            num_classes (int): number of classes used for conditioning
            use_vae (bool): if a VAE is used before flow matching
            vae_encoder_weights (str): path to pre-trained VAE encoder weights
            vae_decoder_weights (str): path to pre-trained VAE decoder weights
            arch_type (str): UNet or ViT
            arch_cfg (dict): backbone parameters, including class_dropout_prob
        """

        super().__init__()

        if img_shape is None or len(img_shape) != 3 or any(s <= 0 for s in img_shape):
            raise ValueError("img_shape must contain positive channel, height, width values")
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("timesteps must be a positive integer")
        if num_classes is None or num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.num_classes = num_classes
        self.timesteps = timesteps
        self.dt = 1.0 / timesteps

        # if we should consider latent DDPM
        self.use_vae = use_vae
        if use_vae:
            if vae_encoder_weights is None or vae_decoder_weights is None:
                raise ValueError("Both VAE encoder and decoder weights are required")
            self.vae = TAESD(
                encoder_path=vae_encoder_weights,
                decoder_path=vae_decoder_weights
            )
            # freeze the encoder / decoder
            for param in self.vae.parameters():
                param.requires_grad = False

        # if VAE is considered, input / output dim will be the dim of latent
        if use_vae:
            if img_shape[0] != 3 or any(s % self.vae.patch_size for s in img_shape[1:]):
                raise ValueError("TAESD requires RGB images with dimensions divisible by its patch size")
            self.img_shape = [
                self.vae.latent_channels,
                img_shape[1]//self.vae.patch_size,
                img_shape[2]//self.vae.patch_size
            ]
        else:
            self.img_shape = img_shape

        self.model = build_arch(
            arch_type, arch_cfg, img_shape=self.img_shape, num_classes=num_classes
        )

    # compute the simplified loss
    def compute_loss(self, x_start, label, noise=None):
        """
        Compute the rectified flow loss for training the model.
        Note: t in the range of [0, 1]
        """

        """
        Fill in the missing code here. See Rectified Flow paper.
        """
        # latent modeling: encode images in [0, 1] into TAESD latents
        # (4 x H/8 x W/8, roughly unit variance); the encoder stays frozen
        if self.use_vae:
            with torch.no_grad():
                x_start = self.vae.encode(x_start)

        batch_size = x_start.shape[0]

        # z: pure random static, same shape as the real images
        if noise is None:
            noise = torch.randn_like(x_start)

        # t: a random point on the noise -> image line, uniform in [0, 1]
        # (0 = all noise, 1 = clean image); one t per image in the batch
        t = torch.rand(batch_size, device=x_start.device)
        # reshape to (B, 1, 1, 1) so it multiplies every pixel of its image
        t_ = t.view(batch_size, *([1] * (x_start.dim() - 1)))

        # Eq. (1): x_t = (1 - t) * z + t * x_data  (a blend of noise and image)
        x_t = (1.0 - t_) * noise + t_ * x_start

        # Eq. (2): the true velocity along the straight path is x_data - z
        target = x_start - noise

        # network's guess of the velocity, given the blend, class label and time
        pred = self.model(x_t, label, t)

        # Eq. (3): mean squared error between the guess and the true velocity
        loss = F.mse_loss(pred, target)
        return loss

    @torch.no_grad()
    @sampling_mode
    def generate(
        self, labels, cfg=False, cfg_scale=1.0, cfg_start_t=0.0, cfg_method="cfg"
    ):
        """
        Sample from FM; cfg_start_t is progress from noise (0) to image (1).
        cfg_method defaults to standard CFG ("cfg").
        """
        device = next(self.model.parameters()).device
        labels = labels.to(device=device, dtype=torch.long)
        if cfg_method not in ("cfg", "cfg++"):
            raise ValueError("cfg_method must be 'cfg' or 'cfg++'")
        # shape of the results
        shape = [len(labels)] + list(self.img_shape)
        # start from pure noise (for each example in the batch)
        imgs = torch.randn(shape, device=device)

        """
        Fill in the missing code here. See Rectified Flow paper.
        """
        # Eq. (5), forward Euler: walk from t = 0 (noise) to t = 1 (image)
        # in self.timesteps equal steps of size self.dt
        # (classifier-free guidance, the bonus, is not implemented yet)
        for i in range(self.timesteps):
            t = torch.full(
                (len(labels),), i * self.dt, device=device, dtype=imgs.dtype
            )
            # which direction (and how fast) to move from here
            velocity = self.model(imgs, labels, t)
            # take one small step in that direction
            imgs = imgs + self.dt * velocity

        # latent modeling: decode generated latents back to images in [0, 1]
        if self.use_vae:
            imgs = self.vae.decode(imgs)

        # postprocessing the images
        imgs = self.postprocess(imgs)
        return imgs

    @torch.no_grad()
    def postprocess(self, imgs):
        """
        Postprocess the sampled images (e.g., normalization / clipping)
        """
        if self.use_vae:
            # already in range, clip the pixel values
            imgs.clamp_(min=0.0, max=1.0)
        else:
            # clip the pixel values within range
            imgs.clamp_(min=-1.0, max=1.0)
            # mapping to range [0, 1]
            imgs = (imgs + 1.0) * 0.5
        return imgs
