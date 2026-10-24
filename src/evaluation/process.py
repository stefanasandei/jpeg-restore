"""Persistent, environment-isolated workers for official baseline implementations."""

import json
import select
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .adapters import Adapter


class ProcessAdapter(Adapter):
    def __init__(self, spec, device, dtype):
        super().__init__(spec, device, dtype)
        repo = Path(spec.get("repo") or "")
        if not spec.get("repo") or not repo.is_dir():
            raise ValueError(
                f"set eval.model_specs.{spec['kind']}.repo to the official checkout"
            )
        self.trained = True
        self.work = TemporaryDirectory(prefix="jpeg-eval-")
        self.directory = Path(self.work.name)
        self.log = (self.directory / "worker.log").open("w+")
        self.process = None
        self.timeout = float(spec.get("timeout", 1800))
        config = self.directory / "config.json"
        config.write_text(
            json.dumps({"spec": spec, "device": str(device), "dtype": dtype,
                        "cpu_threads": torch.get_num_threads(),
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                        "tf32_cudnn": torch.backends.cudnn.allow_tf32})
        )
        worker = Path(__file__).with_name("worker.py")
        try:
            self.process = subprocess.Popen(
                [spec.get("python") or sys.executable, str(worker), str(config)],
                cwd=repo,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
            ready = self._receive()
            self.parameters = ready["parameters"]
            self.nfe = ready["nfe"]
            self.worker_metadata = ready["environment"]
            self.timing_scope = ready["timing_scope"]
            self.worker_weight_dtypes = ready["weight_dtypes"]
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError(f"{self.spec['kind']} worker exceeded {self.timeout:g}s")
        line = self.process.stdout.readline()
        if not line:
            self.log.flush()
            detail = (self.directory / "worker.log").read_text()[-6000:]
            raise RuntimeError(f"{self.spec['kind']} worker exited: {detail}")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def _request(self, operation, item, **kwargs):
        np.save(self.directory / "input.npy", item.rgb.detach().cpu().numpy())
        jpeg_path = self.directory / "input.jpg"
        if item.jpeg is not None:
            jpeg_path.write_bytes(item.jpeg)
        elif jpeg_path.exists():
            jpeg_path.unlink()
        request = {"operation": operation, "directory": str(self.directory), **kwargs}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self._receive()
        self.nfe = response.get("nfe", self.nfe)
        return response

    def prepare(self, item):
        return item

    def restore(self, prepared, seed):
        self._request("restore", prepared, seed=seed)
        return torch.from_numpy(
            np.load(self.directory / "output.npy", allow_pickle=False)
        ).to(self.device)

    def restore_protocol(self, prepared, seed):
        # The isolated worker applies the common protocol exactly once.
        return self.restore(prepared, seed)

    def benchmark(self, item, warmup, iterations, seed):
        return self._request(
            "benchmark", item, warmup=warmup, iterations=iterations, seed=seed
        )

    def weight_dtypes(self):
        return self.worker_weight_dtypes

    def parameter_count(self):
        return self.parameters

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(json.dumps({"operation": "close"}) + "\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.log.close()
        self.work.cleanup()
