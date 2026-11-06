import torch

from typing import Any
from pathlib import Path
from sbervae.lib.models.sdvae.sdvae import SDVAE
from dit_updates.vae.adapters.base import (VAEPreprocessor,
                                           VAEAdapter,
                                           load_latent_stats)
from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import (LatentNormalizationType,
                                                  TorchLatentNormalizer)
from dit_updates.vae.adapters.wan_mil import (WANYuv2YuvPreprocessor,
                                              WANYuv2RgbPreprocessor)


def load_sdvae_state_dict(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """
    Load SDVAE weights from either a raw ``SDVAE`` state dict or a SberVAE
    training checkpoint (``{"model": SDVAEModel.state_dict(), ...}``).

    Args:
        checkpoint (str | Path): Checkpoint path.

    Returns:
        dict[str, torch.Tensor]: State dict ready for ``SDVAE.load_state_dict``.
    """
    state_dict = torch.load(checkpoint, map_location="cpu")
    for key in ("model", "model_state_dict"):
        if key in state_dict and isinstance(state_dict[key], dict):
            state_dict = state_dict[key]
            break
    # SDVAEModel wraps SDVAE as `_model`, DDP adds `module.`.
    prefixes = ("module.", "_model.")
    new_state_dict = {}
    for k, v in state_dict.items():
        for prefix in prefixes:
            if k.startswith(prefix):
                k = k[len(prefix):]
        new_state_dict[k] = v
    return new_state_dict


class SDVAEAdapterBase(VAEAdapter):
    """
    Internal SDVAE adapter base class.
    """

    def __init__(self,
                 name: str,
                 checkpoint: str | Path,
                 latent_norm_type: LatentNormalizationType | str,
                 latent_stats_mean: list[float],
                 latent_stats_std: list[float],
                 prerpocessor_cls: type[VAEPreprocessor],
                 sdvae_kwargs: dict[str, Any],
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SDVAEAdapterBase.

        Args:
            name (str): Adapter/model name.
            checkpoint (str | Path): VAE checkpoint path, raw state dict or SberVAE training checkpoint.
            latent_norm_type (LatentNormalizationType | str): Type of latent normalization.
            latent_stats_mean (list[float]): Per-channel latent mean.
            latent_stats_std (list[float]): Per-channel latent std.
            prerpocessor_cls (type[VAEPreprocessor]): Preprocessor class matching the model input space.
            sdvae_kwargs (dict[str, Any]): Extra kwargs for ``SDVAE`` (e.g. ``output_act``).
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        model = SDVAE(**sdvae_kwargs)
        super(SDVAEAdapterBase, self).__init__(name=name, n_channels=model.z_dim)
        latent_norm_type = LatentNormalizationType(latent_norm_type)
        self._preprocessor_cls = prerpocessor_cls

        checkpoint = resolve_path(checkpoint, "model")
        # model.load_state_dict(load_sdvae_state_dict(checkpoint))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model = model.eval()
        model = model.requires_grad_(False)
        model = model.to(device)
        self._model = model

        mean = torch.tensor(latent_stats_mean, dtype=dtype, device=device)
        std = torch.tensor(latent_stats_std, dtype=dtype, device=device)
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
        Create and return the preprocessor instance.

        Returns:
            VAEPreprocessor: Preprocessor object.
        """
        return self._preprocessor_cls()

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

        info = {}

        return images, info


class SDVAEYuv2YuvAdapter(SDVAEAdapterBase):
    """
    Internal SDVAE YUV2YUV adapter implementation.
    Trained with `sbervae/configs/sdvae/sdvae-yuv2yuv.yaml`: YUV 420d input,
    decoder outputs Y in [0, 1] (sigmoid) and U, V in physical ranges (tanh).
    """

    def __init__(self,
                 name: str = "sdvae-mil-yuv2yuv",
                 checkpoint: str | Path = "SD-VAE-1.5-yuv2yuv/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SDVAEYuv2YuvAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "sdvae-mil-yuv2yuv".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-SDVAE-YUV2YUV/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization (path to ".json" stats file, or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(SDVAEYuv2YuvAdapter, latent_stats, 4)

        super(SDVAEYuv2YuvAdapter, self).__init__(name=name,
                                                  checkpoint=checkpoint,
                                                  latent_norm_type=latent_norm_type,
                                                  latent_stats_mean=mean,
                                                  latent_stats_std=std,
                                                  prerpocessor_cls=WANYuv2YuvPreprocessor,
                                                  sdvae_kwargs={"output_act": "yuv2yuv"},
                                                  device=device,
                                                  dtype=dtype)


class SDVAEYuv2RgbAdapter(SDVAEAdapterBase):
    """
    Internal SDVAE YUV2RGB adapter implementation.
    Trained with `sbervae/configs/sdvae/sdvae-yuv2rgb.yaml`: YUV 420d input,
    decoder outputs RGB images in [0, 1] (sigmoid).
    """

    def __init__(self,
                 name: str = "sdvae-mil-yuv2rgb",
                 checkpoint: str | Path = "SD-VAE-1.5-yuv2rgb/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SDVAEYuv2RgbAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "sdvae-mil-yuv2rgb".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "SD-VAE-1.5-yuv2rgb/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization (path to ".json" stats file, or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(SDVAEYuv2RgbAdapter, latent_stats, 4)

        super(SDVAEYuv2RgbAdapter, self).__init__(name=name,
                                                  checkpoint=checkpoint,
                                                  latent_norm_type=latent_norm_type,
                                                  latent_stats_mean=mean,
                                                  latent_stats_std=std,
                                                  prerpocessor_cls=WANYuv2RgbPreprocessor,
                                                  sdvae_kwargs={"output_act": "yuv2rgb"},
                                                  device=device,
                                                  dtype=dtype)
