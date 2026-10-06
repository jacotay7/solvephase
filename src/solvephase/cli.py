"""Command-line interface: ``solvephase info`` and ``solvephase retrieve``."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .__about__ import __version__

__all__ = ["main"]


def _info(_: argparse.Namespace) -> int:
    from .backend import gpu_available

    print(f"solvephase {__version__}")
    print(f"python     {platform.python_version()} ({platform.platform()})")
    print(f"numpy      {np.__version__}")
    import scipy

    print(f"scipy      {scipy.__version__}")
    for name in ("aobasis", "cupy", "pyturb", "getframes", "hcipy"):
        try:
            module = __import__(name)
            print(f"{name:<10} {getattr(module, '__version__', 'installed')}")
        except ImportError:
            print(f"{name:<10} not installed")
    if gpu_available():
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        print(f"gpu        {name} (CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()})")
    else:
        print("gpu        not available (install solvephase[cuda12] or [cuda13])")
    return 0


def _load_images(path: str) -> np.ndarray:
    source = Path(path)
    if source.suffix == ".npy":
        return np.asarray(np.load(source), dtype=np.float64)
    if source.suffix == ".npz":
        archive = np.load(source)
        return np.asarray(archive[archive.files[0]], dtype=np.float64)
    if source.suffix.lower() in (".fits", ".fit", ".fts"):
        try:
            from astropy.io import fits
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise SystemExit("reading FITS needs astropy: pip install 'solvephase[fits]'") from exc
        return np.asarray(fits.getdata(source), dtype=np.float64)
    raise SystemExit(f"unsupported image file {source.suffix!r}; use .npy, .npz or .fits")


def _retrieve(args: argparse.Namespace) -> int:
    from .api import retrieve
    from .pupil import Pupil

    images = _load_images(args.images)
    if args.pupil in ("keck", "jwst", "vlt", "hst"):
        pupil = getattr(Pupil, args.pupil)(args.pupil_pixels)
    else:
        pupil = Pupil.circular(args.pupil_pixels, args.diameter, obscuration=args.obscuration)
    result = retrieve(
        images,
        pupil,
        args.wavelength,
        sampling=args.sampling,
        diversity=args.defocus,
        basis=args.modes,
        method=args.method,
        loss=args.loss,
        read_noise=args.read_noise,
        device=args.device,
    )
    print(result.summary())
    out = result.save(args.output)
    print(f"saved {out}")
    if args.json:
        payload = {
            "method": result.method,
            "loss": result.loss,
            "rms_m": result.rms(),
            "coefficients_m": None if result.coefficients is None else result.coefficients.tolist(),
            "labels": None if result.basis is None else result.basis.labels,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``solvephase`` command."""
    parser = argparse.ArgumentParser(prog="solvephase", description="Fast phase retrieval.")
    parser.add_argument("--version", action="version", version=f"solvephase {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="show versions and the GPU")
    info.set_defaults(func=_info)

    ret = sub.add_parser("retrieve", help="retrieve a wavefront from focal-plane images")
    ret.add_argument("images", help="(K, my, mx) or (my, mx) images: .npy, .npz or .fits")
    ret.add_argument("--wavelength", type=float, required=True, help="metres")
    ret.add_argument("--sampling", type=float, required=True, help="pixels per lambda/D")
    ret.add_argument(
        "--defocus", type=float, nargs="+", default=None, help="defocus RMS per image in metres"
    )
    ret.add_argument("--pupil", default="circular", help="circular, keck, jwst, vlt or hst")
    ret.add_argument("--pupil-pixels", type=int, default=128)
    ret.add_argument("--diameter", type=float, default=1.0, help="metres (circular pupil)")
    ret.add_argument("--obscuration", type=float, default=0.0)
    ret.add_argument("--modes", type=int, default=36, help="number of Zernike modes")
    ret.add_argument("--method", default="auto")
    ret.add_argument("--loss", default="poisson")
    ret.add_argument("--read-noise", type=float, default=0.0, help="electrons")
    ret.add_argument("--device", default="cpu")
    ret.add_argument("--output", default="solvephase-result.npz")
    ret.add_argument("--json", default=None, help="also write a JSON summary here")
    ret.set_defaults(func=_retrieve)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
