from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.cuda.amp as amp
import torchvision.transforms as T

from typing import Any
from pathlib import Path
from dit_updates.vae.adapters.base import (VAEPreprocessor, 
                                           VAEAdapter, 
                                           load_latent_stats)
from dit_updates.vae.models.wan import _video_vae
from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import (LatentNormalizationType,
                                                  TorchLatentNormalizer)
from dit_updates.vae.models.sdvae import SDVAE


class SDVAEOfficialPreprocessor(VAEPreprocessor):
    """
    Official preprocessor for the SDVAE model.
    """

    def __init__(self):
        """
        Initialize the preprocessor with standard normalization.
        """
        super(SDVAEOfficialPreprocessor, self).__init__()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        return 2. * image - 1.

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse normalization of the image.

        Args:
            image (torch.Tensor): Normalized image tensor.

        Returns:
            torch.Tensor: De-normalized image tensor.
        """
        return 0.5 * image + 0.5


class SDVAEOfficialAdapter(VAEAdapter):
    """
    SDVAE official adapter implementation.
    """

    def __init__(self,
                 name: str = "sdvae_official",
                 checkpoint: str | Path = "SD-VAE-1.5-ft-mse-official/sdvae_1.5_ft_mse_official.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = "imagenet2012",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANOfficialAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "sdvae_official".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "SD-VAE-1.5-ft-mse-official/sdvae_1.5_ft_mse_official.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "official", or None). Defaults to "imagenet2012".
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        super().__init__(name=name, n_channels=4)
        latent_norm_type = LatentNormalizationType(latent_norm_type)

        checkpoint = resolve_path(checkpoint, "model")
        model = SDVAE()
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model = model.eval()
        model = model.requires_grad_(False)
        model = model.to(device)
        self._model = model

        mean, std = load_latent_stats(SDVAEOfficialAdapter, latent_stats, model.z_dim)

        mean = torch.tensor(mean, dtype=dtype, device=device)
        std = torch.tensor(std, dtype=dtype, device=device)
        self._latent_normalizer = latent_norm_type.make_torch(
            mean, std, device)

        self._dtype = dtype

    @property
    def latent_normalizer(self) -> TorchLatentNormalizer:
        """
        Return the latent normalizer instance.

        Returns:
            TorchLatentNormalizer: The latent normalizer.
        """
        return self._latent_normalizer

    def create_preprocessor(self) -> VAEPreprocessor:
        """
        Create and return the WANOfficialPreprocessor instance.

        Returns:
            WANOfficialPreprocessor: Preprocessor object.
        """
        return SDVAEOfficialPreprocessor()

    @torch.inference_mode()
    def encode(self,
               images: torch.Tensor,
               normalize: bool = True,
               sample: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Encode images into the latent space of the VAE.

        Args:
            images (torch.Tensor): Images to be encoded, (B, C, H, W) or (C, H, W).
            normalize (bool, optional): Normalize latents using the latent normalizer. Defaults to True.
            sample (bool, optional): Sample from posterior latent distribution. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Encoded latents and info dictionary.
        """
        shape = images.shape
        if len(shape) == 3:  # (C, H, w)
            images = images.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        distribution = self._model.encode(images)
        mean = distribution.mean
        logvar = distribution.logvar

        if sample:
            latents = distribution.sample()
        else:
            latents = distribution.mode()

        if normalize:
            latents = self._latent_normalizer.normalize(latents)

        if single:
            latents = latents.squeeze(0)  # (C, H, W)
            mean = mean.squeeze(0)
            logvar = logvar.squeeze(0)

        info = {
            "raw_distribution": DiagonalGaussianDistribution(mean, logvar)
        }

        return latents, info

    @torch.inference_mode()
    def decode(self,
               latents: torch.Tensor,
               denormalize: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Decode latents into images using the VAE decoder.

        Args:
            latents (torch.Tensor): Latents to decode. Shape can be (B, C, H, W) or (C, H, W).
            denormalize (bool, optional): Denormalize latents before decoding. Defaults to True.

        Returns:
            tuple[torch.Tensor, dict[str, Any]]: Decoded images and info dictionary.
        """
        shape = latents.shape
        if len(shape) == 3:  # (C, H, w)
            latents = latents.unsqueeze(0)  # (B, C, H, W)
            single = True
        else:  # (B, C, H, W)
            single = False

        if denormalize:
            latents = self._latent_normalizer.denormalize(latents)

        images = self._model.decode(latents)

        if single:
            images = images.squeeze(0)  # (C, H, W)
        images = images.clamp(-1, 1)  # From official code

        info = {}

        return images, info
