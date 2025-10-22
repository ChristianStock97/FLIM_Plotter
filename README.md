# FLIM_Plotter

Lightweight FLIM (Fluorescence Lifetime Imaging Microscopy) viewer and simple GPU-accelerated fitter built around Napari and Numba/CUDA.

This repository contains a small GUI tool (`FLim_plotter.py`) to load FLIM stacks (TIFF), perform a per-pixel exponential fit on the fluorescence decay using CUDA-accelerated kernels, and visualize results (intensity, tau maps and RGB-mapped images) in Napari.

## Features

- Load interleaved FLIM TIFF stacks (3D or 4D) using `tiffile`.
- GPU-accelerated operations implemented with Numba/CUDA for dilation, mean/intensity calculation and per-pixel fitting.
- Visualize results interactively with Napari (Intensity, Tau, Dilated/TimeStack and RGB composite).
- Simple Qt-based control panel to adjust kernel size, threshold, time-binning and tau-range.

## Requirements

This project was developed for Python and requires a CUDA-capable GPU for the GPU kernels. Key dependencies are listed in `requirements.txt`:

- napari[all]
- numpy
- numba[cuda]
- tiffile

Install the dependencies in a virtual environment, for example (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Notes:
- `numba[cuda]` and a working NVIDIA CUDA toolkit + drivers are required to run the CUDA kernels.
- Napari installation may pull a number of GUI dependencies; consult Napari docs if you run into issues.

## Quick start

Run the GUI script directly:

```powershell
python FLim_plotter.py
```

This will open a Napari viewer window with a dockable control panel. Use the "load Stack" button to select a TIFF stack and then press "Run Fit" to compute maps. The control panel exposes sliders for kernel size, threshold, tau range and time-binning.

## Configuration

Default starting values are read from `config.ini` (created automatically if missing). Example values:

```
[Starting Parameters]
kernel = 3
threshold = 1000
mintau = 8
maxtau = 18
startsample = 9
time_bin = 1
```

The GUI will update and save the starting parameters back to `config.ini` when you run a fit.

## Key files

- `FLim_plotter.py` – main Qt/Napari GUI and orchestration for loading data and launching GPU kernels.
- `global_fit.py` – CUDA kernels (dilation, averaging, per-pixel fit) implemented with Numba.
- `TO_HSV.py` – conversion utilities that map tau + intensity to RGB and additional CPU helper functions.
- `requirements.txt` – Python dependencies.
- `config.ini` – saved GUI start parameters (auto-created).

## Development notes

- The code assumes TIFF stacks where time is either the first or second dimension depending on interleaving. See the loader in `FLim_plotter.py` for details.
- Error handling is minimal; running on systems without CUDA will raise errors when kernels are invoked.

If you want to run parts of the pipeline on CPU only, you can adapt the GPU calls in `global_fit.py` and `FLim_plotter.py` (this is left as an exercise for advanced users).

## License

This project is released under the MIT License — see `LICENSE` for details.

## Acknowledgements

Built with Napari and Numba. Thanks to the open-source ecosystem.
