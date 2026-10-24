"""Fetch pinned trained baselines and build the official coefficient reader locally."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    "FBCNN": ("https://github.com/jiaxi-jiang/FBCNN.git", "54d1831927506b3247e2d4d245abb4f4dab1a1cd"),
    "JDEC": ("https://github.com/WooKyoungHan/JDEC.git", "e9fe8354754dd062dbf1defebb220e290507edd2"),
    "PromptCIR-NTIRE24": ("https://github.com/lbc12345/PromptCIR-NTIRE24.git", "3506b1aa0e6588fdf4e0b8345d90679adf1e2220"),
}


def run(*command, **kwargs):
    subprocess.run(list(command), check=True, **kwargs)


def prepare_metrics(base):
    """Populate isolated caches with verified metric bytes before GPU execution."""
    from huggingface_hub import hf_hub_download
    import importlib.util
    import urllib.request
    archive = ROOT / "artifacts/matched-24-oct/assets.json"
    identities = json.loads(archive.read_text())["assets"]
    acquisition = []
    from datetime import datetime, timezone
    for item in identities:
        name = Path(item["file"]).name
        if name not in {"alexnet-owt-7be5be79.pth", "vgg16-397923af.pth",
                        "musiq_koniq_ckpt-e95806b9.pth", "ckpt_koniq10k.pt"}:
            continue
        pyiqa = name in {"musiq_koniq_ckpt-e95806b9.pth", "ckpt_koniq10k.pt"}
        target = base / "torch-cache/hub" / ("pyiqa" if pyiqa else "checkpoints") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc).isoformat()
        operation = "already_present"
        source = Path(item["file"])
        url = ("https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/0df2df423c65f6a64209309695f3845727431027/" + name) if pyiqa else item["source_url"]
        if not target.exists():
            if source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"]:
                shutil.copy2(source, target)
                operation = "verified_local_copy"
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                urllib.request.urlretrieve(url, temporary)
                temporary.replace(target)
                operation = "downloaded"
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise RuntimeError(f"metric weight bytes differ: {target}")
        acquisition.append({"file": str(target), "operation": operation, "source_url": url,
                            "sha256": digest, "started_utc": started,
                            "completed_utc": datetime.now(timezone.utc).isoformat()})
    cache = base / "hub-cache"
    repo = "timm/vit_base_patch8_224.augreg2_in21k_ft_in1k"
    revision = "907a22023d1c918aae6d7f340275616a25f7f2ff"
    path = Path(hf_hub_download(repo, "model.safetensors", revision=revision, cache_dir=cache))
    expected = next(item["sha256"] for item in identities if item.get("hub_revision") == revision)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError("timm metric backbone differs from archived weights")
    # This cache belongs exclusively to this protocol. Timm requests 'main';
    # bind that local reference to the pinned snapshot before offline execution.
    reference = cache / ("models--" + repo.replace("/", "--")) / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    reference.write_text(revision)
    package = Path(importlib.util.find_spec("torchmetrics").origin).parent
    for item in identities:
        if item["acquisition"] == "installed_package":
            relative = item["file"].split("/torchmetrics/", 1)[1]
            if hashlib.sha256((package / relative).read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(f"TorchMetrics bundled weights differ: {relative}")
    (base / "preparation.json").write_text(json.dumps(acquisition, indent=2) + "\n")


def prepare_codiff(storage):
    """Prepare the trained diffusion baseline without using the root disk."""
    from datetime import datetime, timezone
    import gdown
    storage = Path(storage).resolve()
    for name in ("models/codiff", "vendor", "cache/hf", "cache/pip", "tmp"):
        (storage / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(storage / "cache/hf"),
                      HF_HUB_CACHE=str(storage / "cache/hf/hub"),
                      PIP_CACHE_DIR=str(storage / "cache/pip"),
                      TMPDIR=str(storage / "tmp"))
    repo = storage / "vendor/CODiff"
    revision = "86e2f0499456298613472fa164417d1cce0a0cb7"
    if not repo.exists():
        run("git", "clone", "https://github.com/jp-guo/CODiff.git", str(repo))
    current = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if current != revision:
        if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
            raise RuntimeError("refusing to replace a modified CODiff checkout")
        run("git", "-C", str(repo), "checkout", revision)
    patch = ROOT / "artifacts/full-image-codiff/codiff-fp32.patch"
    applied = subprocess.run(["git", "-C", str(repo), "apply", "--reverse", "--check", str(patch)], capture_output=True).returncode == 0
    if not applied:
        run("git", "-C", str(repo), "apply", str(patch))
    base = storage / "models/codiff"
    records = []
    for filename, identifier in [("codiff.pkl", "1SHZQs8fu3K419jlENoY8qdTKbjAOL0i7"),
                                  ("cave.pth", "1SZo5UMSMEcYy4Wr-OIQgTk8hUXiFMIbK")]:
        target = base / filename
        started = datetime.now(timezone.utc).isoformat()
        operation = "already_present" if target.exists() else "downloaded"
        if not target.exists():
            gdown.download(id=identifier, output=str(target), quiet=False)
        records.append(dict(file=str(target), operation=operation, started_utc=started,
                            completed_utc=datetime.now(timezone.utc).isoformat()))
    run("hf", "download", "Manojb/stable-diffusion-2-1-base",
        "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json",
        "--revision", "0094d483a120f3f33dafbd187ea4aa60d10de75c", "--local-dir", str(base / "sd21"))
    assets = json.loads((ROOT / "artifacts/full-image-codiff/codiff-assets.json").read_text())
    for asset in [assets[0], assets[1], *assets[2]["files"]]:
        target = storage / asset["file"]
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != asset["sha256"]:
            raise RuntimeError(f"CODiff asset hash differs: {target}")
    from importlib.metadata import distributions
    dependencies = storage / "vendor/codiff-deps"
    pinned = {"diffusers": "0.25.0", "peft": "0.9.0", "huggingface-hub": "0.25.0",
              "transformers": "4.48.3", "tokenizers": "0.21.4"}
    installed = {d.metadata["Name"].lower().replace("_", "-"): d.version
                 for d in distributions(path=[str(dependencies)])}
    if any(installed.get(name) != version for name, version in pinned.items()):
        run(sys.executable, "-m", "pip", "install", "--target", str(dependencies), "--upgrade",
            "--no-deps", "--only-binary=:all:", *(f"{name}=={version}" for name, version in pinned.items()))
    (base / "preparation.json").write_text(json.dumps(records, indent=2) + "\n")
    print("Trained CODiff/CaVE, SD2.1 components, and isolated dependencies are ready:", storage)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codiff-only", action="store_true")
    parser.add_argument("--storage-root", default="/mnt/projects/jpeg-restoration")
    args = parser.parse_args()
    if args.codiff_only:
        prepare_codiff(args.storage_root)
        return
    os.chdir(ROOT)
    storage = Path(args.storage_root).resolve()
    for directory in ("models/matched", "cache/hf", "cache/pip", "tmp"):
        (storage / directory).mkdir(parents=True, exist_ok=True)
    os.environ.update(HF_HOME=str(storage / "cache/hf"),
                      HF_HUB_CACHE=str(storage / "cache/hf/hub"),
                      PIP_CACHE_DIR=str(storage / "cache/pip"), TMPDIR=str(storage / "tmp"))
    for name, (url, revision) in REPOS.items():
        target = Path("external") / name
        if not (target / ".git").exists():
            target.mkdir(parents=True, exist_ok=True)
            run("git", "init", "-q", str(target))
            run("git", "-C", str(target), "remote", "add", "origin", url)
        current = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                          text=True, stderr=subprocess.DEVNULL).strip() if (target / ".git/HEAD").exists() and (target / "README.md").exists() else None
        if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            raise RuntimeError(f"refusing modified upstream checkout: {target}")
        if current != revision:
            run("git", "-C", str(target), "fetch", "--depth", "1", "origin", revision)
            run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")
    base = storage / "models/matched"
    base.mkdir(parents=True, exist_ok=True)
    fbcnn = base / "fbcnn_color.pth"
    if not fbcnn.exists():
        run("curl", "-L", "--fail", "--retry", "2", "https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth", "-o", str(fbcnn))
    for identifier, path in [
        ("1_TLoGHLBtdbPdApRmI0y0YW2nEqnO8Dr", base / "jdec.pth"),
        ("1Ll673246UclKHy_MwEIfcTOtW658UZ3Q", base / "promptcir/checkpoint/model_best.pth"),
    ]:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            run("gdown", "https://drive.google.com/uc?id=" + identifier, "-O", str(path))
    run("hf", "download", "asandeistefan/jpeg-restore-meanflows", "meanflow_jpeg_qf5_20.pth",
        "--revision", "33315831b6b342d3b8ddbdc70aa11cb84e55c66b", "--local-dir", str(base / "meanflow"))
    run("hf", "download", "openai/clip-vit-base-patch16", "pytorch_model.bin", "config.json",
        "preprocessor_config.json", "tokenizer_config.json", "special_tokens_map.json", "merges.txt", "vocab.json",
        "--revision", "57c216476eefef5ab752ec549e440a49ae4ae5f3", "--local-dir", str(base / "clip-vit-base-patch16"))
    run(sys.executable, "-m", "pip", "install", "--target", str(base / "python-deps"),
        "--no-deps", "tensorboardX==2.6.5", "jpegio==0.2.8")
    os.environ["TORCH_EXTENSIONS_DIR"] = str((base / "dct-extension").resolve())
    os.environ.setdefault("MAX_JOBS", "2")
    os.environ.setdefault("CXX", shutil.which("clang++") or "c++")
    from torch.utils.cpp_extension import load
    load(name="dct_manip", sources=[str(ROOT / "external/JDEC/dct_manip/dct_manip.cpp")],
         extra_cflags=["-O1", "-std=c++20"], extra_ldflags=["-ljpeg"], verbose=False)
    archived = ROOT / "artifacts/matched-24-oct/assets.json"
    if archived.exists():
        for item in json.loads(archived.read_text())["assets"]:
            relative = Path(item["file"])
            path = base / relative.relative_to("checkpoints/matched") if str(relative).startswith("checkpoints/matched/") else relative
            if str(relative).startswith("checkpoints/") and path.is_file() and path.suffix != ".so":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != item["sha256"]:
                    raise RuntimeError(f"downloaded asset differs from published bytes: {path}")
    prepare_metrics(base)
    print("Pinned baseline sources, trained weights, metric assets, and coefficient reader are ready.")


if __name__ == "__main__":
    main()
