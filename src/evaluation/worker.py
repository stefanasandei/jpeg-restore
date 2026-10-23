"""Worker bootstrap. Official repositories get import priority over local models."""

import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

# A worker starts in the official repository; never import local `models` here.
sys.path.insert(0, str(Path.cwd()))
sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    import numpy as np
    import torch

    from evaluation.adapters import seed_everything, validate_output
    from evaluation.data import EvaluationInput
    from evaluation.external import create_external
    from evaluation.reporting import environment_metadata

    protocol = sys.stdout

    def send(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()

    try:
        config = json.loads(Path(sys.argv[1]).read_text())
        # Some official repositories use the current CUDA device implicitly.
        device = torch.device(config["device"])
        if device.type == "cuda":
            torch.cuda.set_device(device)
        seed_everything(0)
        with redirect_stdout(sys.stderr):
            adapter = create_external(config["spec"], config["device"], config["dtype"])
        send(
            {
                "parameters": adapter.parameter_count(),
                "nfe": adapter.nfe,
                "environment": environment_metadata(device),
                "timing_scope": adapter.timing_scope,
                "weight_dtypes": adapter.weight_dtypes(),
            }
        )
    except Exception:  # noqa: BLE001 - serialize upstream failures across the process boundary.
        send({"error": traceback.format_exc()})
        return

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request["operation"] == "close":
                with redirect_stdout(sys.stderr):
                    adapter.close()
                return
            directory = Path(request["directory"])
            rgb = torch.from_numpy(np.load(directory / "input.npy", allow_pickle=False))
            jpeg_path = directory / "input.jpg"
            item = EvaluationInput(
                rgb, jpeg_path.read_bytes() if jpeg_path.exists() else None
            )
            with redirect_stdout(sys.stderr):
                if request["operation"] == "benchmark":
                    response = adapter.benchmark(
                        item, request["warmup"], request["iterations"], request["seed"]
                    )
                else:
                    seed_everything(request["seed"])
                    prepared = adapter.prepare(item)
                    with torch.inference_mode(), adapter.precision_context():
                        output = adapter.restore(prepared, request["seed"])
                    output = validate_output(output, item.rgb.shape)
                    np.save(directory / "output.npy", output.cpu().numpy())
                    response = {"status": "ok", "nfe": adapter.nfe}
            send(response)
        except Exception:  # noqa: BLE001 - serialize upstream failures across the process boundary.
            send({"error": traceback.format_exc()})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
