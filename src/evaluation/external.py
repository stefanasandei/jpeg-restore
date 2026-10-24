"""Thin adapters to the official APIs; imported only inside isolated workers."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F

from .adapters import Adapter, seed_everything
from .data import jpeg_size

SOURCES = {
    "fbcnn": "https://github.com/jiaxi-jiang/FBCNN",
    "jdec": "https://github.com/WooKyoungHan/JDEC",
    "promptcir": "https://github.com/lbc12345/PromptCIR-NTIRE24",
    "diffbir": "https://github.com/XPixelGroup/DiffBIR",
    "supir": "https://github.com/Fanghua-Yu/SUPIR",
}


def checkpoint_path(spec, key="checkpoint"):
    path = spec.get(key)
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            f"set {key} to a local official checkpoint for {spec['kind']}"
        )
    return path


def load_weights(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "params_ema", "params"):
        if key in state:
            state = state[key]
            break
    return {key.removeprefix("module."): value for key, value in state.items()}


class FBCNNAdapter(Adapter):
    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        from models.network_fbcnn import FBCNN

        self.model = FBCNN(
            in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R"
        )
        self.model.load_state_dict(load_weights(checkpoint_path(spec)), strict=True)
        self.model = self.model.eval().to(self.device)
        self.nfe = 1

    def restore(self, prepared, seed):
        return self.model(prepared)[0]


class JDECAdapter(Adapter):
    timing_scope = "jpeg_coefficients_to_rgb"

    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        import dct_manip

        import models

        if self.device.type != "cuda":
            raise ValueError("the official JDEC model uses CUDA internally")
        self.dm = dct_manip
        state = torch.load(checkpoint_path(spec), map_location="cpu", weights_only=True)
        encoder = state["model"]["args"]["encoder_spec"]
        if encoder["name"] == "swinv2_rgb_nomore_deepir_rgbver":
            # The released checkpoint retains the old registry label; the
            # official train_JDEC.yaml names this factory swinv2_group_embedded.
            # Arguments and every learned tensor are unchanged; loading is strict.
            encoder["name"] = "swinv2_group_embedded"
        self.model = models.make(state["model"], load_sd=True).eval().to(self.device)
        self.nfe = 1
        self.work = TemporaryDirectory(prefix="jdec-coefficients-")
        self.jpeg_path = Path(self.work.name) / "input.jpg"

    def prepare(self, item):
        if item.jpeg is None:
            raise ValueError("JDEC needs original JPEG bytes")
        if item.rgb.shape[0] != 1:
            raise ValueError("JDEC coefficient evaluation supports batch size 1")
        self.jpeg_path.write_bytes(item.jpeg)
        return item.jpeg

    def restore(self, prepared, seed):
        # Coefficient parsing and transfers are part of the system path. The
        # input JPEG file is staged outside timing, just like RGB input loading.
        coefficients = self.dm.read_coefficients(str(self.jpeg_path))
        q_y, q_c = coefficients[1][0], coefficients[1][1]
        y = (coefficients[2] * q_y).clamp(-1024, 1016)
        c = (coefficients[3] * q_c).clamp(-1024, 1016)
        quantization = torch.stack([q_y, q_c])
        normalize = lambda value: (value + 1024) / 2040 * 2 - 1
        inputs = [
            normalize(value).unsqueeze(0).to(self.device)
            for value in (y, c, quantization)
        ]
        restored = self.model(*inputs) + 0.5
        height, width = jpeg_size(prepared)
        return restored[..., :height, :width]

    def restore_protocol(self, prepared, seed):
        tiling = self.spec.get("common_tiling")
        if not tiling:
            return self.restore(prepared, seed)
        from .tiling import stitch

        height, width = jpeg_size(prepared)
        if height % 32 or width % 32:
            raise ValueError("matched JDEC inputs must share the 32-pixel JPEG padding")
        coefficients = self.dm.read_coefficients(str(self.jpeg_path))
        q_y, q_c = coefficients[1][0], coefficients[1][1]
        y = (coefficients[2] * q_y).clamp(-1024, 1016)
        c = (coefficients[3] * q_c).clamp(-1024, 1016)
        normalize = lambda value: (value + 1024) / 2040 * 2 - 1
        y, c, q = [normalize(value).unsqueeze(0).to(self.device)
                   for value in (y, c, torch.stack([q_y, q_c]))]
        def run(top, left, h, w, index):
            if any(value % 32 for value in (top, left, h, w)):
                raise ValueError("matched coefficient tiles must align to 32 pixels")
            yt = y[:, :, top//8:(top+h)//8, left//8:(left+w)//8]
            ct = c[:, :, top//16:(top+h)//16, left//16:(left+w)//16]
            return self.model(yt, ct, q) + 0.5
        output, count = stitch(height, width, self.device, run, **tiling)
        self.nfe = count
        return output

    def close(self):
        super().close()
        self.work.cleanup()


class PromptCIRAdapter(Adapter):
    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        from model import MyModel

        self.model = MyModel(decoder=True)
        self.model.load_state_dict(load_weights(checkpoint_path(spec)), strict=True)
        self.model = self.model.eval().to(self.device)
        # Explicit full-image, single-model path. The challenge release uses
        # tiling, two models, and transpose ensembling; those change the protocol.
        self.nfe = 1

    def restore(self, prepared, seed):
        height, width = prepared.shape[-2:]
        padding = (0, (-width) % 32, 0, (-height) % 32)
        restored = self.model(F.pad(prepared, padding, mode="replicate"))
        return restored[..., :height, :width]


class SUPIRAdapter(Adapter):
    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        if self.device.type != "cuda":
            raise ValueError("the official SUPIR implementation requires CUDA")
        from SUPIR.util import create_SUPIR_model

        config = spec.get("config")
        if not config or not Path(config).is_file():
            raise FileNotFoundError(
                "set SUPIR config to a YAML with local SDXL and SUPIR weights"
            )
        self.model = create_SUPIR_model(config, SUPIR_sign=spec.get("variant", "Q"))
        self.model.ae_dtype = (
            torch.float32 if self.dtype == torch.float32 else torch.bfloat16
        )
        self.model.model.dtype = self.dtype
        self.model = self.model.eval().to(self.device)
        self.calls = 0
        self.model.model.register_forward_pre_hook(self._count)

    def _count(self, *_):
        self.calls += 1

    def precision_context(self):
        # SUPIR selects VAE and denoiser precision internally.
        return nullcontext()

    def restore(self, prepared, seed):
        height, width = prepared.shape[-2:]
        image = F.pad(
            prepared * 2 - 1, (0, (-width) % 64, 0, (-height) % 64), mode="replicate"
        )
        self.calls = 0
        output = self.model.batchify_sample(
            image,
            [self.spec.get("prompt", "")] * image.shape[0],
            num_steps=self.spec.get("steps", 50),
            seed=seed,
            num_samples=1,
            **self.spec.get("sampling", {}),
        )
        self.nfe = self.calls
        return ((output + 1) / 2)[..., :height, :width]


class DiffBIRAdapter(Adapter):
    timing_scope = "host_rgb_through_official_pipeline"

    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        if self.device.type != "cuda":
            raise ValueError("the official DiffBIR inference loop requires CUDA")
        # Use the official parser's defaults without consuming the harness CLI.
        module_spec = importlib.util.spec_from_file_location(
            "diffbir_official_cli", Path.cwd() / "inference.py"
        )
        cli = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(cli)
        original_argv = sys.argv
        try:
            sys.argv = ["inference.py", "--input", ".", "--output", "."]
            args = cli.parse_args()
        finally:
            sys.argv = original_argv
        args.device = str(self.device)
        args.precision = {
            torch.float32: "fp32",
            torch.float16: "fp16",
            torch.bfloat16: "bf16",
        }[self.dtype]
        args.version = spec.get("version", "custom")
        args.upscale = 1
        # A custom model is required to reproduce the paper's DiffBIR* row.
        if args.version == "custom":
            args.ckpt = checkpoint_path(spec)
            args.train_cfg = spec.get("config")
            if not args.train_cfg or not Path(args.train_cfg).is_file():
                raise FileNotFoundError(
                    "DiffBIR custom evaluation needs its training config"
                )
            from diffbir.inference import CustomInferenceLoop

            self.loop = CustomInferenceLoop(args)
        else:
            from diffbir.inference import BSRInferenceLoop

            self.loop = BSRInferenceLoop(args)
        self.model = torch.nn.ModuleList([self.loop.cleaner, self.loop.cldm])
        self.args = args
        self.calls = 0
        # Count denoiser calls rather than inferring NFE from solver step count.
        self.loop.cldm.register_forward_pre_hook(self._count)

    def _count(self, *_):
        self.calls += 1

    def prepare(self, item):
        return item.rgb.permute(0, 2, 3, 1).mul(255).round().byte().numpy()

    def restore(self, prepared, seed):
        seed_everything(seed)
        args = self.args
        options = {
            "steps": self.spec.get("steps", 50),
            "strength": args.strength,
            "cleaner_tiled": False,
            "cleaner_tile_size": args.cleaner_tile_size,
            "cleaner_tile_stride": args.cleaner_tile_stride,
            "vae_encoder_tiled": False,
            "vae_encoder_tile_size": args.vae_encoder_tile_size,
            "vae_decoder_tiled": False,
            "vae_decoder_tile_size": args.vae_decoder_tile_size,
            "cldm_tiled": False,
            "cldm_tile_size": args.cldm_tile_size,
            "cldm_tile_stride": args.cldm_tile_stride,
            "pos_prompt": self.spec.get("prompt", ""),
            "neg_prompt": args.neg_prompt,
            "cfg_scale": args.cfg_scale,
            "start_point_type": args.start_point_type,
            "sampler_type": self.spec.get("solver", "ddim"),
            "noise_aug": args.noise_aug,
            "rescale_cfg": args.rescale_cfg,
            "s_churn": args.s_churn,
            "s_tmin": args.s_tmin,
            "s_tmax": args.s_tmax,
            "s_noise": args.s_noise,
            "eta": args.eta,
            "order": args.order,
        }
        options.update(self.spec.get("sampling", {}))
        self.calls = 0
        output = self.loop.pipeline.run(prepared, **options)
        self.nfe = self.calls
        return (
            torch.from_numpy(output).permute(0, 3, 1, 2).to(self.device).float() / 255
        )


def create_external(spec, device, dtype):
    classes = {
        "fbcnn": FBCNNAdapter,
        "jdec": JDECAdapter,
        "promptcir": PromptCIRAdapter,
        "diffbir": DiffBIRAdapter,
        "supir": SUPIRAdapter,
    }
    return classes[spec["kind"]](spec, device, dtype)
