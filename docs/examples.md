# Examples

The scripts in
[`examples/`](https://github.com/jacotay7/solvephase/tree/main/examples) are
headless and deterministic. Each takes `--output DIR` (for its figures) and
`--device cpu|gpu|auto`. CI runs every script in a reduced mode
(`SOLVEPHASE_QUICK=1`).

| Script | Shows |
|---|---|
| `01_ncpa_calibration.py` | NCPA calibration on a VLT-like pupil from two getframes detector frames with saturation masking, using `retrieve` |
| `02_broadband_undersampled.py` | Hubble-like broadband, undersampled data (about 1 px per λ/D) with pixel integration; exact broadband vs monochromatic models |
| `03_segment_phasing.py` | Piston/tip/tilt of all 36 segments of a Keck-like primary from two defocused images |
| `04_extended_scene_diversity.py` | Gonsalves/Paxman phase diversity on an unknown extended scene: wavefront and object |
| `05_focal_plane_wfs.py` | LIFT against its Cramér–Rao bound, and a Fast & Furious closed loop |
| `06_cdi_multistart.py` | CDI with an autocorrelation support, shrinkwrap and 16 batched random starts |
| `07_generic_coded_diffraction.py` | Wirtinger-flow family on coded-diffraction measurements |
| `08_tie_microscopy.py` | Transport-of-intensity phase imaging of a weakly absorbing sample |
| `showcase.py` | Renders the README animation |

```bash
python examples/01_ncpa_calibration.py --output figures --device auto
```
