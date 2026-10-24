import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "model"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_output(output, shape):
    if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(shape):
        raise ValueError(
            f"expected RGB output {tuple(shape)}, got {getattr(output, 'shape', type(output))}"
        )
    if not torch.isfinite(output).all():
        raise FloatingPointError("restoration contains non-finite pixels")
    return output.float().clamp(0, 1)


class Adapter:
    """prepare() stays outside timing; restore() includes the full restoration path."""

    scope = "system"
    timing_scope = "resident_rgb_to_rgb"

    def __init__(self, spec, device, dtype):
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = DTYPES[dtype]
        self.model = None
        self.nfe = None
        boundary = spec.get("benchmark_input", "resident_rgb")
        if boundary not in ("resident_rgb", "cached_jpeg_file"):
            raise ValueError("unknown benchmark_input boundary")
        if boundary == "cached_jpeg_file":
            self.timing_scope = "cached_jpeg_file_to_rgb"

    def prepare(self, item):
        return item.rgb.to(self.device)

    def restore(self, prepared, seed):
        raise NotImplementedError

    def restore_protocol(self, prepared, seed):
        tiling = self.spec.get("common_tiling")
        if not tiling:
            return self.restore(prepared, seed)
        from .tiling import stitch

        if prepared.shape[0] != 1:
            raise ValueError("the matched tiling protocol requires batch one")
        if not hasattr(self, "_base_nfe"):
            self._base_nfe = self.nfe
        def run(top, left, height, width, index):
            return self.restore(prepared[..., top:top+height, left:left+width],
                                (seed + index) % 2**32)
        output, count = stitch(*prepared.shape[-2:], self.device, run, **tiling)
        self.nfe = self._base_nfe * count
        return output

    def precision_context(self):
        return (
            torch.autocast(self.device.type, dtype=self.dtype)
            if self.dtype != torch.float32
            else nullcontext()
        )

    def weight_dtypes(self):
        return (
            sorted({str(p.dtype) for p in self.model.parameters()})
            if self.model is not None
            else []
        )

    def parameter_count(self):
        return (
            sum(p.numel() for p in self.model.parameters())
            if self.model is not None
            else 0
        )

    def benchmark(self, item, warmup, iterations, seed):
        from .performance import measure

        return measure(self, item, warmup, iterations, seed)

    def close(self):
        self.model = None


class JPEGAdapter(Adapter):
    scope = "identity"

    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        self.nfe = 0

    def restore(self, prepared, seed):
        return prepared


class NativeAdapter(Adapter):
    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from checkpoint import model_state

        architecture = spec.get("architecture")
        if "model" in spec:
            model_cfg = OmegaConf.create(spec["model"])
        else:
            model_cfg = OmegaConf.load(CONFIG_DIR / f"{architecture}.yaml").model
        model_cfg = OmegaConf.merge(model_cfg, spec.get("overrides", {}))
        representation = spec.get("representation", "pixel")
        levels = spec.get("wavelet_levels", 2)
        if representation == "haar":
            channels = 3 * 4**levels
            input_name, output_name = {
                "restormer": ("inp_channels", "out_channels"),
                "fbcnn": ("in_nc", "out_nc"),
            }.get(architecture, ("in_channels", "out_channels"))
            model_cfg[input_name], model_cfg[output_name] = channels, channels
            if architecture == "fbcnn":
                model_cfg.clamp_output = False

        self.resolved_model_config = OmegaConf.to_container(model_cfg, resolve=True)
        self.model = instantiate(model_cfg)
        if representation == "haar":
            from models.wavelet import HaarWaveletRestoration

            self.model = HaarWaveletRestoration(self.model, levels=levels)
        checkpoint = spec.get("checkpoint")
        self.trained = bool(checkpoint)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            weights = model_state(state)
            expected = set(self.model.state_dict())
            # Published inference exports omit the outer training wrapper.
            # Only normalize a prefix when the complete key set matches.
            candidates = [weights]
            if weights and all(key.startswith("module.") for key in weights):
                candidates.append(
                    {
                        key.removeprefix("module."): value
                        for key, value in weights.items()
                    }
                )
            candidates.extend(
                {f"model.{key}": value for key, value in candidate.items()}
                for candidate in list(candidates)
            )
            matching = next(
                (candidate for candidate in candidates if set(candidate) == expected),
                None,
            )
            if matching is None:
                raise ValueError(
                    f"checkpoint keys do not match {architecture or model_cfg.get('_target_')}; "
                    "use its training architecture/config or a compatible published export"
                )
            self.model.load_state_dict(matching, strict=True)
        self.model = self.model.eval().to(device=self.device, dtype=self.dtype)
        self.scope = spec.get("scope", "system")
        steps = spec.get("steps")
        if steps is not None:
            if not hasattr(self.model, "sample_steps"):
                raise ValueError("steps require a sampling model")
            self.model.sample_steps = steps
        solver = spec.get("solver")
        if solver:
            if solver not in ("euler", "heun") or not hasattr(
                self.model, "sampling_method"
            ):
                raise ValueError(
                    "solver requires RectifiedFlow and must be euler or heun"
                )
            self.model.sampling_method = solver
        steps = getattr(self.model, "sample_steps", 1)
        self.nfe = steps * (
            2 if getattr(self.model, "sampling_method", None) == "heun" else 1
        )
        if spec.get("compile", False):
            core = getattr(self.model, "model", self.model)
            core.forward = torch.compile(
                core.forward,
                backend="inductor",
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    def restore(self, prepared, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if hasattr(self.model, "sample"):
            output = self.model.sample(prepared, generator=generator)
        else:
            output = self.model(prepared)
        return output[0] if isinstance(output, tuple) else output


def build_adapter(spec, device, dtype):
    kind = spec.get("kind", "native")
    if kind == "jpeg":
        return JPEGAdapter(spec, device, dtype)
    if kind == "native":
        return NativeAdapter(spec, device, dtype)
    if kind in ("fbcnn", "jdec", "promptcir", "diffbir", "supir"):
        from .process import ProcessAdapter

        return ProcessAdapter(spec, device, dtype)
    if kind == "custom":
        from hydra.utils import get_object

        return get_object(spec["factory"])(spec, device, dtype)
    raise ValueError(f"unknown model adapter {kind!r}")
