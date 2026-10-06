# Installation

solvephase needs Python 3.10 or newer.

```bash
pip install solvephase                 # CPU: NumPy, SciPy, aobasis
pip install "solvephase[cuda12]"       # + CuPy for CUDA 12.x drivers
pip install "solvephase[cuda13]"       # + CuPy for CUDA 13.x drivers
pip install "solvephase[plot]"         # + matplotlib for Result.plot()
pip install "solvephase[interop]"      # + pyturb and getframes for realistic data
pip install "solvephase[fits]"         # + astropy to read FITS in the CLI
```

Pick the CUDA extra that matches your driver (`nvidia-smi` prints the CUDA
version it supports). The extras install CuPy with the CUDA headers it needs
to compile kernels; if you install CuPy yourself and the first GPU call fails
with *"Failed to find CUDA headers"*, install `cupy-cuda12x[ctk]` (or
`cupy-cuda13x[ctk]`).

Check what solvephase sees:

```bash
solvephase info
```

## From source

```bash
git clone https://github.com/jacotay7/solvephase
cd solvephase
pip install -e ".[dev,docs,interop,validation]"
python -m pytest -q
```

See [Contributing](contributing.md) for the full quality gate.
