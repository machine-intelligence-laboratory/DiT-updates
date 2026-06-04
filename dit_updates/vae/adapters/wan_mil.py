import torch
import torchvision.transforms as T
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.nn as nn

from typing import Any
from pathlib import Path
from kornia.color.yuv import rgb_to_yuv420, yuv420_to_rgb
from sbervae.lib.models import WanVAEModel
from sbervae.lib.models.wan_vae.wan_vae import WanVAE_FCS
from dit_updates.vae.adapters.base import (VAEPreprocessor, 
                                           VAEAdapter, 
                                           load_latent_stats)
from dit_updates.utils.files import resolve_path
from dit_updates.vae.models.distributions import DiagonalGaussianDistribution
from dit_updates.vae.models.normalization import (LatentNormalizationType,
                                                  TorchLatentNormalizer)
from dit_updates.vae.adapters.wan_official import WANOfficialPreprocessor
from dit_updates.vae.models.wan_split import WanImageYUVSplitVAE


class WANYuv2RgbPreprocessor(VAEPreprocessor):
    """
    Preprocess for internal WAN 2.1 YUV2RGB model.
    """

    def __init__(self):
        """
        Initialize the preprocessor with standard normalization.
        """
        super(WANYuv2RgbPreprocessor, self).__init__()
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor [0, 1] in RGB space.

        Returns:
            torch.Tensor: Image tensor in YUV space.
        """
        # This transform is aligned with one from SberVAE and current
        # baseline YUV2RGB model, which uses YUV 420d transform.
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y, uv = rgb_to_yuv420(image)
        uv = F.interpolate(uv, scale_factor=2, mode='bilinear')
        yuv = torch.cat([y, uv], dim=1)
        if not batched:
            yuv = yuv.squeeze(0)
        return yuv

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of the YUV2RGB model.
        It does nothing since output is already in RGB space.

        Args:
            image (torch.Tensor): Image tensor in RGB space.

        Returns:
            torch.Tensor: Same image tensor.
        """
        return image


class WANYuv2YuvPreprocessor(VAEPreprocessor):
    """
    Preprocess for internal WAN 2.1 YUV2YUV model.
    """

    def __init__(self):
        """
        Initialize the preprocessor with standard normalization.
        """
        super(WANYuv2YuvPreprocessor, self).__init__()
        
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to the image.

        Args:
            image (torch.Tensor): Input image tensor [0, 1] in RGB space.

        Returns:
            torch.Tensor: Image tensor in YUV space.
        """
        # This transform is aligned with one from SberVAE and current
        # baseline YUV2RGB model, which uses YUV 420d transform.
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y, uv = rgb_to_yuv420(image)
        uv = F.interpolate(uv, scale_factor=2, mode='bilinear')
        yuv = torch.cat([y, uv], dim=1)
        if not batched:
            yuv = yuv.squeeze(0)
        return yuv

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of the YUV2RGB model.
        Does inverse YUV to RGB transform.

        Args:
            image (torch.Tensor): Image tensor in RGB space.

        Returns:
            torch.Tensor: Same image tensor.
        """
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        y = image[:, :1, :, :]
        uv = image[:, 1:, :, :]
        uv = F.interpolate(uv, scale_factor=0.5, mode='bilinear')
        rgb = yuv420_to_rgb(y, uv)
        if not batched:
            rgb = rgb.squeeze(0)
        return rgb


class DummyPreprocessor(VAEPreprocessor):
    """
    Dummy preprocessor that does nothing.
    """

    def __init__(self):
        """
        Initialize the dummy preprocessor.
        """
        super(DummyPreprocessor, self).__init__()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply dummy preprocessor.
        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Same image tensor.
        """
        return image

    def inverse(self, image: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform of the dummy preprocessor.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Same image tensor.
        """
        return image


class WANAdapterBase(VAEAdapter):
    """
    Internal WAN 2.1 adapter base class.
    """

    def __init__(self,
                 model_cls: type[nn.Module],
                 name: str,
                 checkpoint: str | Path,
                 latent_norm_type: LatentNormalizationType | str,
                 latent_stats_mean: list[float],
                 latent_stats_std: list[float],
                 prerpocessor_cls: type[VAEPreprocessor],
                 wan_kwargs: dict[str, Any],
                 temporal_dim: bool = True,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANOfficialAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan_2.1_official".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "Wan2.1-T2V-14B/Wan2.1_VAE.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "official", or None). Defaults to "imagenet2012".
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        super(WANAdapterBase, self).__init__(name=name, n_channels=16)
        latent_norm_type = LatentNormalizationType(latent_norm_type)
        self._preprocessor_cls = prerpocessor_cls

        checkpoint = resolve_path(checkpoint, "model")

        model = model_cls(pretrained_path=str(checkpoint),
                          **wan_kwargs)
        model = model.eval()
        model = model.to(device)
        self._model = model

        mean = torch.tensor(latent_stats_mean, dtype=dtype, device=device)
        std = torch.tensor(latent_stats_std, dtype=dtype, device=device)
        self._latent_normalizer = latent_norm_type.make_torch(
            mean, std, device)

        self._temporal_dim = temporal_dim
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
        
        if self._temporal_dim:
            images = images.unsqueeze(2)  # (B, C, T, H, W) for model

        distribution = self._model.encode(images)
        mean = distribution.mean
        logvar = distribution.logvar
        if sample:
            latents = distribution.sample()
        else:
            latents = distribution.mode()
        if self._temporal_dim:
            latents = latents.squeeze(2)  # (B, C, H, W)
            mean = mean.squeeze(2)
            logvar = logvar.squeeze(2)

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

        if self._temporal_dim:
            latents = latents.unsqueeze(2)  # (B, C, T, H, W) for model

        images = self._model.decode(latents)
        if self._temporal_dim:
            images = images.squeeze(2)  # (B, C, H, W)
        if single:
            images = images.squeeze(0)  # (C, H, W)

        info = {}

        return images, info


class WANYuv2RgbAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2RGB adapter implementation.
    """

    _IMAGENET_2012_200_MEAN = [
        -7.196443038992584e-05,
        -0.19263234734535217,
        -0.0060315364971756935,
        -0.008871454745531082,
        0.012358635663986206,
        -0.1800689995288849,
        -0.028683781623840332,
        0.09497839212417603,
        0.014750940725207329,
        0.0012209581909701228,
        -0.013892349787056446,
        0.0063743325881659985,
        0.03340122103691101,
        0.11512420326471329,
        0.1839529573917389,
        -0.06042119115591049
    ],

    _IMAGENET_2012_200_STD = [
        0.0006977790035307407,
        0.6388314962387085,
        0.9277921915054321,
        0.7258686423301697,
        0.9773375391960144,
        0.8358289003372192,
        0.7998051643371582,
        1.0100041627883911,
        0.6669394373893738,
        1.0282669067382812,
        0.7665238380432129,
        0.7367441058158875,
        0.8998191356658936,
        0.7038764357566833,
        0.6587797999382019,
        0.6429328322410583
    ]

    _IMAGENET_2012_MEAN = [
        -7.286853360710666e-05,
        -0.18078672885894775,
        -0.008212699554860592,
        -0.007905763573944569,
        0.019507741555571556,
        -0.16465966403484344,
        -0.02668576128780842,
        0.08449669182300568,
        0.008322713896632195,
        0.008041778579354286,
        -0.012736196629703045,
        0.00617537135258317,
        0.03087935410439968,
        0.10986445099115372,
        0.17370596528053284,
        -0.05156862363219261
    ]

    _IMAGENET_2012_STD = [
        0.0006974305724725127,
        0.6393429636955261,
        0.9233031868934631,
        0.7281091213226318,
        0.9659150838851929,
        0.8390931487083435,
        0.7966273427009583,
        1.003061056137085,
        0.6728984117507935,
        1.0245347023010254,
        0.7747780084609985,
        0.7395362854003906,
        0.9027074575424194,
        0.7035373449325562,
        0.6566562056541443,
        0.6456491947174072
    ]

    def __init__(self,
                 name: str = "wan-mil-yuv2rgb",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2RGB/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2RgbAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2rgb".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2RGB/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANYuv2RgbAdapter, latent_stats, 16)

        super(WANYuv2RgbAdapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2RgbPreprocessor,
                                                wan_kwargs={"output_act": "yuv2rgb"},
                                                device=device, 
                                                dtype=dtype)


class WANYuv2RgbStage1Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2RGB Stage 1 adapter implementation.
    """

    def __init__(self,
                 name: str = "wan-mil-yuv2rgb-stage1",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2RGB-Stage1/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2RgbStage1Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2rgb-stage1".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2RGB-Stage1/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANYuv2RgbStage1Adapter, latent_stats, 16)

        super(WANYuv2RgbStage1Adapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2RgbPreprocessor,
                                                wan_kwargs={"output_act": "yuv2rgb"},
                                                device=device, 
                                                dtype=dtype)


class WANYuv2YuvAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2YUV adapter implementation.
    """

    def __init__(self,
                 name: str = "wan-mil-yuv2yuv",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2YUV/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2YuvAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2yuv".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2YUV/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANYuv2YuvAdapter, latent_stats, 16)

        super(WANYuv2YuvAdapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2YuvPreprocessor,
                                                wan_kwargs={"output_act": "yuv2yuv"},
                                                device=device, 
                                                dtype=dtype)


class WANYuv2YuvStage1Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2YUV Stage 1 adapter implementation.
    """

    def __init__(self,
                 name: str = "wan-mil-yuv2yuv-stage1",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2YUV-Stage1/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2YuvStage1Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2yuv-stage1".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2YUV-Stage1/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANYuv2YuvStage1Adapter, latent_stats, 16)

        super(WANYuv2YuvStage1Adapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2YuvPreprocessor,
                                                wan_kwargs={"output_act": "yuv2yuv"},
                                                device=device, 
                                                dtype=dtype)


class WANRgb2RgbAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 RGB2RGB adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.00038634706288576126,
        -0.06125892698764801,
        -0.3927463889122009,
        0.5678133368492126,
        -0.4044269323348999,
        -0.060593269765377045,
        0.3320796489715576,
        -0.1419297754764557,
        -0.18930397927761078,
        0.06352389603853226,
        0.011293075978755951,
        0.630565881729126,
        -0.06407170742750168,
        -0.16139191389083862,
        0.06914106756448746,
        0.0072408802807331085
    ]

    _IMAGENET_2012_STD = [
        0.004230488557368517,
        1.4905462265014648,
        1.6916476488113403,
        1.63594388961792,
        1.9967753887176514,
        1.5547175407409668,
        1.4185155630111694,
        1.4980685710906982,
        1.4175094366073608,
        1.6830899715423584,
        1.582936406135559,
        1.4860095977783203,
        1.570357322692871,
        1.536580204963684,
        1.6044707298278809,
        1.4705582857131958
    ]

    def __init__(self,
                 name: str = "wan-mil-rgb2rgb",
                 checkpoint: str | Path = "MIL-Wan2.1-RGB2RGB/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANRgb2RgbAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-rgb2rgb".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-RGB2RGB/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANRgb2RgbAdapter, latent_stats, 16)

        super(WANRgb2RgbAdapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANOfficialPreprocessor,
                                                wan_kwargs={},
                                                device=device, 
                                                dtype=dtype)


class WANRgb2RgbStage1Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 RGB2RGB Stage 1 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.0002701055782381445,
        -0.008249727077782154,
        -0.005353990010917187,
        0.02219092845916748,
        0.028756937012076378,
        -0.009144437499344349,
        0.0065582818351686,
        -0.009602917358279228,
        0.013471106998622417,
        -0.021000217646360397,
        0.005977471359074116,
        -0.007104683201760054,
        -0.0024013062939047813,
        0.0077857039868831635,
        0.009034544229507446,
        -0.0018136906437575817
    ]

    _IMAGENET_2012_STD = [
        0.0007571529131382704,
        0.9397653937339783,
        1.005461573600769,
        0.9541499018669128,
        0.96479332447052,
        0.9787197709083557,
        0.9988489151000977,
        0.9475454688072205,
        0.9903336763381958,
        0.9909127354621887,
        0.944443941116333,
        0.9630937576293945,
        0.9618390202522278,
        0.9751954078674316,
        1.0314021110534668,
        0.9647769927978516
    ]

    def __init__(self,
                 name: str = "wan-mil-rgb2rgb-stage1",
                 checkpoint: str | Path = "MIL-Wan2.1-RGB2RGB-Stage1/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANRgb2RgbStage1Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-rgb2rgb-stage1".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-RGB2RGB-Stage1/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANRgb2RgbStage1Adapter, latent_stats, 16)

        super(WANRgb2RgbStage1Adapter, self).__init__(model_cls=WanVAEModel,
                                                      name=name,
                                                      checkpoint=checkpoint,
                                                      latent_norm_type=latent_norm_type,
                                                      latent_stats_mean=mean,
                                                      latent_stats_std=std,
                                                      prerpocessor_cls=WANOfficialPreprocessor,
                                                      wan_kwargs={},
                                                      device=device, 
                                                      dtype=dtype)


class WANFCSAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 FCS adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.026889808475971222,
        0.021572163328528404,
        -0.10603587329387665,
        -0.04300852492451668,
        -0.0044104657135903835,
        0.08977996557950974,
        -0.06413924694061279,
        -0.009045019745826721,
        0.10603518038988113,
        0.11772821843624115,
        0.12388269603252411,
        -0.42700937390327454,
        0.002769499784335494,
        -0.3348850607872009,
        -0.11531104892492294,
        -0.089259572327137
    ]

    _IMAGENET_2012_STD = [
        0.9723424315452576,
        0.98272705078125,
        0.996949315071106,
        0.9888678789138794,
        0.9848778247833252,
        0.9801453948020935,
        0.9851140379905701,
        1.0364147424697876,
        0.9703274965286255,
        0.9923965930938721,
        0.9730899930000305,
        0.8813202977180481,
        0.9827848672866821,
        0.9316514730453491,
        0.9687932133674622,
        0.9763295650482178
    ]

    def __init__(self,
                 name: str = "wan-mil-fcs",
                 checkpoint: str | Path = "MIL-Wan2.1-FreqReg/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANFCSAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-fcs".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-FreqReg/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANFCSAdapter, latent_stats, 16)

        super(WANFCSAdapter, self).__init__(model_cls=WanVAE_FCS,
                                            name=name,
                                            checkpoint=checkpoint,
                                            latent_norm_type=latent_norm_type,
                                            latent_stats_mean=mean,
                                            latent_stats_std=std,
                                            prerpocessor_cls=WANOfficialPreprocessor,
                                            wan_kwargs={"frequency_separator": "default"},
                                            device=device, 
                                            dtype=dtype)


class WANFCSStage1Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 FCS Stage 1 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.02687048353254795,
        0.021635930985212326,
        -0.10609440505504608,
        -0.04298165813088417,
        -0.004310328979045153,
        0.08974995464086533,
        -0.06409131735563278,
        -0.008961755782365799,
        0.10602729022502899,
        0.11782369017601013,
        0.1238691508769989,
        -0.4270171523094177,
        0.002716794842854142,
        -0.33489570021629333,
        -0.1153683215379715,
        -0.0892302468419075
    ]

    _IMAGENET_2012_STD = [
        0.9723082780838013,
        0.9827009439468384,
        0.9969533681869507,
        0.9888582229614258,
        0.9848828315734863,
        0.9801485538482666,
        0.9850901961326599,
        1.0365149974822998,
        0.9703118205070496,
        0.9923986792564392,
        0.973099946975708,
        0.8813082575798035,
        0.98276287317276,
        0.9316624402999878,
        0.9687702655792236,
        0.9763883948326111
    ]

    def __init__(self,
                 name: str = "wan-mil-fcs-stage1",
                 checkpoint: str | Path = "MIL-Wan2.1-FreqReg-Stage1/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANFCSStage1Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-fcs-stage1".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-FreqReg-Stage1/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANFCSStage1Adapter, latent_stats, 16)

        super(WANFCSStage1Adapter, self).__init__(model_cls=WanVAE_FCS,
                                            name=name,
                                            checkpoint=checkpoint,
                                            latent_norm_type=latent_norm_type,
                                            latent_stats_mean=mean,
                                            latent_stats_std=std,
                                            prerpocessor_cls=WANOfficialPreprocessor,
                                            wan_kwargs={"frequency_separator": "default"},
                                            device=device, 
                                            dtype=dtype)


class WANSplitAttn12to4Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split Attention 12to4 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.003839731914922595,
        0.005159264896064997,
        0.009019199758768082,
        -0.0007820131722837687,
        0.0011651229579001665,
        0.0063963960856199265,
        0.003206053515896201,
        -0.010293500497937202,
        -0.006534602027386427,
        -0.004016075283288956,
        0.003994189202785492,
        -0.004355896729975939,
        -0.006619404535740614,
        0.0004192973137833178,
        0.005864075850695372,
        0.0017565203597769141
    ]

    _IMAGENET_2012_STD = [
        0.9926429986953735,
        0.8986800909042358,
        0.962837278842926,
        0.9001625776290894,
        0.9786242842674255,
        0.9304022789001465,
        0.9178804755210876,
        1.019809365272522,
        0.8938657641410828,
        0.9049807190895081,
        0.9471054077148438,
        0.9187670350074768,
        0.8424476981163025,
        0.9782582521438599,
        0.8591406345367432,
        0.9455100893974304
    ]

    def __init__(self,
                 name: str = "wan-mil-split-attn-12to4",
                 checkpoint: str | Path = "MIL-Wan-Split-Attn-12to4/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplitAttn12to4Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-attn-12to4".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-Attn-12to4/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplitAttn12to4Adapter, latent_stats, 16)

        super(WANSplitAttn12to4Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                       name=name,
                                                       checkpoint=checkpoint,
                                                       latent_norm_type=latent_norm_type,
                                                       latent_stats_mean=mean,
                                                       latent_stats_std=std,
                                                       prerpocessor_cls=DummyPreprocessor,
                                                       wan_kwargs={
                                                            "fusion_type": "attention",
                                                            "fusion_level": "stage1"
                                                        },
                                                       temporal_dim=False,
                                                       device=device, 
                                                       dtype=dtype)


class WANSplitFiLM12to4Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split FiLM 12to4 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        0.014540188014507294,
        0.00233704736456275,
        0.029061334207654,
        -0.00507014337927103,
        0.009950731880962849,
        -0.006688317283987999,
        0.005234121344983578,
        0.0063605643808841705,
        -0.0007046094397082925,
        0.00028301298152655363,
        0.004668115638196468,
        0.005661346949636936,
        -0.00026984387659467757,
        0.006954879034310579,
        0.0011703516356647015,
        0.002588289324194193
    ]

    _IMAGENET_2012_STD = [
        0.966349184513092,
        0.9085988998413086,
        0.9215602874755859,
        0.9815815091133118,
        1.0044193267822266,
        1.0167453289031982,
        0.9498870372772217,
        0.920560896396637,
        0.8935450911521912,
        0.9148443341255188,
        0.8956382274627686,
        0.9284133315086365,
        0.8637551069259644,
        0.9813997149467468,
        0.8028090000152588,
        0.9343475699424744
    ]

    def __init__(self,
                 name: str = "wan-mil-split-film-12to4",
                 checkpoint: str | Path = "MIL-Wan-Split-FiLM-12to4/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplitFiLM12to4Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-film-12to4".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-FiLM-12to4/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplitFiLM12to4Adapter, latent_stats, 16)

        super(WANSplitFiLM12to4Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                       name=name,
                                                       checkpoint=checkpoint,
                                                       latent_norm_type=latent_norm_type,
                                                       latent_stats_mean=mean,
                                                       latent_stats_std=std,
                                                       prerpocessor_cls=DummyPreprocessor,
                                                       wan_kwargs={
                                                            "fusion_type": "film",
                                                            "fusion_level": "stage1"
                                                        },
                                                       temporal_dim=False,
                                                       device=device, 
                                                       dtype=dtype)


class WANSplit12to4Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split 12to4 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        8.244203490903601e-05,
        0.001987516414374113,
        -0.00707614328712225,
        -0.007869222201406956,
        0.002945689717307687,
        0.006798756308853626,
        0.002065645530819893,
        -0.005604919977486134,
        0.01609939895570278,
        0.004559771157801151,
        0.003424854250624776,
        -0.006251154933124781,
        -0.00848422385752201,
        -0.0009118825546465814,
        -0.019699852913618088,
        -0.0009196222526952624
    ]

    _IMAGENET_2012_STD = [
        0.9631742835044861,
        0.9390857815742493,
        0.892459511756897,
        1.018998384475708,
        1.0046337842941284,
        0.9835742115974426,
        0.9405491352081299,
        0.9060302376747131,
        0.8843628764152527,
        0.9149699211120605,
        0.9018592238426208,
        0.9448477029800415,
        0.9129185676574707,
        0.9852203726768494,
        0.8883286714553833,
        0.9477189183235168
    ]

    def __init__(self,
                 name: str = "wan-mil-split-12to4",
                 checkpoint: str | Path = "MIL-Wan-Split-12to4/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplit12to4Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-12to4".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-12to4/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplit12to4Adapter, latent_stats, 16)

        super(WANSplit12to4Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                       name=name,
                                                       checkpoint=checkpoint,
                                                       latent_norm_type=latent_norm_type,
                                                       latent_stats_mean=mean,
                                                       latent_stats_std=std,
                                                       prerpocessor_cls=DummyPreprocessor,
                                                       wan_kwargs={
                                                            "fusion_type": "none",
                                                       },
                                                       temporal_dim=False,
                                                       device=device, 
                                                       dtype=dtype)


class WANYuv2RgbFreqRegAdapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV2RGB FreqReg adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.03650403767824173,
        -0.02067389339208603,
        0.41247713565826416,
        -0.04763864353299141,
        -0.09010378271341324,
        0.06344891339540482,
        -0.1663283258676529,
        0.19659118354320526,
        -0.058062851428985596,
        0.06689545512199402,
        0.15028032660484314,
        -0.0634562149643898,
        -0.5088106989860535,
        0.13859862089157104,
        0.032985761761665344,
        -0.02560049667954445
    ]

    _IMAGENET_2012_STD = [
        0.9531320929527283,
        0.9673222899436951,
        0.8805227875709534,
        0.9694610834121704,
        0.997051477432251,
        0.9803810715675354,
        0.9894546270370483,
        0.9750522375106812,
        0.9716711640357971,
        1.008178949356079,
        0.9507225155830383,
        0.9842593669891357,
        0.8176929950714111,
        0.9655146598815918,
        0.9694461226463318,
        0.9615552425384521
    ]

    def __init__(self,
                 name: str = "wan-mil-yuv2rgb-freqreg",
                 checkpoint: str | Path = "MIL-Wan2.1-YUV2RGB-FreqReg/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANYuv2RgbFreqRegAdapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-yuv2rgb-freqreg".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan2.1-YUV2RGB-FreqReg/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANYuv2RgbFreqRegAdapter, latent_stats, 16)

        super(WANYuv2RgbFreqRegAdapter, self).__init__(model_cls=WanVAEModel,
                                                name=name,
                                                checkpoint=checkpoint,
                                                latent_norm_type=latent_norm_type,
                                                latent_stats_mean=mean,
                                                latent_stats_std=std,
                                                prerpocessor_cls=WANYuv2RgbPreprocessor,
                                                wan_kwargs={"output_act": "yuv2rgb"},
                                                device=device, 
                                                dtype=dtype)


class WANSplitFiLMFreqReg12to4Stage1Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split FiLM 12to4 Stage 1 FreqReg adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        0.039882462471723557,
        -0.4760834574699402,
        -0.2884219288825989,
        -0.049201685935258865,
        -0.049724653363227844,
        -0.1690102368593216,
        0.22590363025665283,
        -0.31897294521331787,
        0.6000936031341553,
        -0.6516937017440796,
        -0.2916269302368164,
        0.3924739360809326,
        -0.10988853126764297,
        -0.08414600044488907,
        0.06488227844238281,
        -0.27731311321258545
    ]

    _IMAGENET_2012_STD = [
        1.0096741914749146,
        0.8466978073120117,
        0.9416480660438538,
        0.9789198637008667,
        0.9967079758644104,
        0.9775936603546143,
        0.953965961933136,
        0.9263364672660828,
        0.7679343223571777,
        0.7025222182273865,
        0.9346325993537903,
        0.9115617275238037,
        0.9735429286956787,
        0.9925159811973572,
        0.9755667448043823,
        0.9443214535713196
    ]

    def __init__(self,
                 name: str = "wan-mil-split-film-freqreg-12to4-stage1",
                 checkpoint: str | Path = "MIL-Wan-Split-FiLM-12to4-FreqReg-Stage1/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplitFiLMFreqReg12to4Stage1Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-film-freqreg-12to4-stage1".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-FiLM-12to4-FreqReg-Stage1/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplitFiLMFreqReg12to4Stage1Adapter, latent_stats, 16)

        super(WANSplitFiLMFreqReg12to4Stage1Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                                    name=name,
                                                                    checkpoint=checkpoint,
                                                                    latent_norm_type=latent_norm_type,
                                                                    latent_stats_mean=mean,
                                                                    latent_stats_std=std,
                                                                    prerpocessor_cls=DummyPreprocessor,
                                                                    wan_kwargs={
                                                                            "fusion_type": "film",
                                                                            "fusion_level": "stage1"
                                                                        },
                                                                    temporal_dim=False,
                                                                    device=device, 
                                                                    dtype=dtype)


class WANSplitFiLM12to4Stage3Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split FiLM 12to4 Stage 3 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        -0.06384941935539246,
        0.000523643393535167,
        -0.018375881016254425,
        0.07969962805509567,
        -0.09246460348367691,
        0.0036549933720380068,
        -0.05953744053840637,
        0.0009438489796593785,
        0.06814761459827423,
        0.009098454378545284,
        0.13947321474552155,
        0.1815461814403534,
        -0.011684859171509743,
        0.009768961928784847,
        -0.0820087268948555,
        0.0036262325011193752
    ]

    _IMAGENET_2012_STD = [
        0.8962666392326355,
        0.7442806363105774,
        0.7768663763999939,
        0.9870763421058655,
        0.6435696482658386,
        1.0197782516479492,
        0.9188668727874756,
        0.7618923783302307,
        0.7478412389755249,
        0.7755031585693359,
        0.7011823654174805,
        0.8091209530830383,
        0.6053034663200378,
        0.9437452554702759,
        0.4902122914791107,
        0.765608549118042
    ]

    def __init__(self,
                 name: str = "wan-mil-split-film-12to4-stage3",
                 checkpoint: str | Path = "MIL-Wan-Split-FiLM-12to4-Stage3/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplitFiLM12to4Stage3Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-film-12to4-stage3".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-FiLM-12to4-Stage3/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplitFiLM12to4Stage3Adapter, latent_stats, 16)

        super(WANSplitFiLM12to4Stage3Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                       name=name,
                                                       checkpoint=checkpoint,
                                                       latent_norm_type=latent_norm_type,
                                                       latent_stats_mean=mean,
                                                       latent_stats_std=std,
                                                       prerpocessor_cls=DummyPreprocessor,
                                                       wan_kwargs={
                                                            "fusion_type": "film",
                                                            "fusion_level": "stage1"
                                                        },
                                                       temporal_dim=False,
                                                       device=device, 
                                                       dtype=dtype)


class WANSplitAttn12to4Stage3Adapter(WANAdapterBase):
    """
    Internal WAN 2.1 YUV Split Attention 12to4 Stage 3 adapter implementation.
    """

    _IMAGENET_2012_MEAN = [
        0.09329797327518463,
        0.011604581028223038,
        -0.08153163641691208,
        -0.19618700444698334,
        0.005008398555219173,
        -0.03407711163163185,
        0.006990359164774418,
        0.003733995370566845,
        0.1466902792453766,
        -0.05536044389009476,
        0.1036577969789505,
        -0.13969597220420837,
        -0.007974648848176003,
        -0.004143456928431988,
        -0.04880741983652115,
        0.009279180318117142
    ]

    _IMAGENET_2012_STD = [
        0.9867312908172607,
        0.7285397052764893,
        0.874036967754364,
        0.7032602429389954,
        0.9258047938346863,
        0.7937049865722656,
        0.7617037892341614,
        1.007412075996399,
        0.6688795685768127,
        0.7299169898033142,
        0.8505851030349731,
        0.7545222043991089,
        0.5326356887817383,
        0.9597623944282532,
        0.5906078815460205,
        0.8044209480285645
    ]


    def __init__(self,
                 name: str = "wan-mil-split-attn-12to4-stage3",
                 checkpoint: str | Path = "MIL-Wan-Split-Attn-12to4-Stage3/model.pth",
                 latent_norm_type: LatentNormalizationType | str = LatentNormalizationType.SCALE,
                 latent_stats: str | None = None,
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the WANSplitAttn12to4Stage3Adapter.

        Args:
            name (str, optional): Adapter/model name. Defaults to "wan-mil-split-attn-12to4-stage3".
            checkpoint (str | Path, optional): VAE checkpoint path. Defaults to "MIL-Wan-Split-Attn-12to4-Stage3/model.pth".
            latent_norm_type (LatentNormalizationType | str, optional): Type of latent normalization. Defaults to LatentNormalizationType.SCALE.
            latent_stats (str | None, optional): Stats to use for normalization ("imagenet2012", "imagenet2012_200", or None). Defaults to None.
            device (str, optional): Device to use. Defaults to "cuda".
            dtype (torch.dtype, optional): Floating point dtype for weights and tensors. Defaults to torch.float32.
        """
        mean, std = load_latent_stats(WANSplitAttn12to4Stage3Adapter, latent_stats, 16)

        super(WANSplitAttn12to4Stage3Adapter, self).__init__(model_cls=WanImageYUVSplitVAE,
                                                       name=name,
                                                       checkpoint=checkpoint,
                                                       latent_norm_type=latent_norm_type,
                                                       latent_stats_mean=mean,
                                                       latent_stats_std=std,
                                                       prerpocessor_cls=DummyPreprocessor,
                                                       wan_kwargs={
                                                            "fusion_type": "attention",
                                                            "fusion_level": "stage1"
                                                        },
                                                       temporal_dim=False,
                                                       device=device, 
                                                       dtype=dtype)

