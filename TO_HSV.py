import numpy as np
import imageio.v3 as iio
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from numba import cuda
import tiffile as tif
import napari
import time
def normalize(img, vmin=None, vmax=None, eps=1e-12):
    """Skaliert Bild linear auf [0,1]."""
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)
    out = (img - vmin) / (vmax - vmin + eps)
    return np.clip(out, 0, 1)

def flim_intensity_to_rgb(flim, intensity, flim_min=None, flim_max=None, 
                          intensity_min=None, intensity_max=None, 
                          gamma=1.0, mask_nan=True):
    """
    Kombiniert FLIM (Hue) und Intensität (Value) zu RGB.
    - flim_*: Bereich der Lebensdauer (z.B. in ns) der auf Hue [0,1] gemappt wird.
    - intensity_*: Bereich der Intensität für Value.
    - gamma: optionales Gamma auf Value (z.B. 0.5 für mehr Kontrast in dunklen Bereichen).
    - mask_nan: NaN/Inf im Output schwärzen.
    """
    # Hue aus FLIM normalisieren
    H = normalize(flim, flim_min, flim_max)

    # Saturation = 1
    S = np.ones_like(H, dtype=float)

    # Value aus Intensität normalisieren + optional Gamma
    V = normalize(intensity, intensity_min, intensity_max)
    if gamma != 1.0:
        V = np.power(V, gamma)

    # HSV -> RGB
    HSV = np.stack([H, S, V], axis=-1)  # (..., 3)
    RGB = hsv_to_rgb(HSV)

    if mask_nan:
        bad = ~np.isfinite(flim) | ~np.isfinite(intensity)
        if np.any(bad):
            RGB[bad] = 0.0
    
    return (RGB * 256).astype(np.uint8)


def old():
    path = "E:/Interleaved FLIM/"
    flim_path = path +  "tau_NoAvr.tif"
    intensity_path = path + "Intensity_NoAvr.tif"
    flim = iio.imread(flim_path).astype(np.float32)
    intensity = iio.imread(intensity_path).astype(np.float32)

    # Wähle sinnvolle Mapping-Bereiche (Beispiel: 1–2.5 ns)
    rgb = flim_intensity_to_rgb(flim, intensity, flim_min=1, flim_max=2.5, gamma=0.9)

    plt.figure(figsize=(6,6))
    plt.imshow(rgb)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    out_path = path + "flim_intensity_100.png"
    iio.imwrite(out_path, (np.clip(rgb, 0, 1)*255).astype(np.uint8))
    print(f"Gespeichert unter: {out_path}")

@cuda.jit
def flim_to_rgb(RGB, tau, intensity, h_min, h_max, max_intensity):
    y = cuda.blockIdx.x
    x = cuda.threadIdx.x

    # convert Data to HSV color
    H = (tau[y, x] - h_min) / (h_max - h_min)
    S = 1.0
    V = intensity[y, x] / max_intensity
    if V > 1.0:
        V = 0
    hi = int(H * 6) % 6
    f = (H * 6) - hi
    p = V * (1 - S)
    q = V * (1 - f * S)
    t = V * (1 - (1 - f) * S)

    V = int(256 * V)
    f = int(256 * f)
    p = int(256 * p)
    q = int(256 * q)
    t = int(256 * t)

    match hi:
        case 0:
            RGB[y, x, 0] = V
            RGB[y, x, 1] = t
            RGB[y, x, 2] = p
        case 1:
            RGB[y, x, 0] = q
            RGB[y, x, 1] = V
            RGB[y, x, 2] = p
        case 2:
            RGB[y, x, 0] = p
            RGB[y, x, 1] = V
            RGB[y, x, 2] = t
        case 3:
            RGB[y, x, 0] = p
            RGB[y, x, 1] = q
            RGB[y, x, 2] = V
        case 4:
            RGB[y, x, 0] = t
            RGB[y, x, 1] = p
            RGB[y, x, 2] = V
        case 5:
            RGB[y, x, 0] = V
            RGB[y, x, 1] = p
            RGB[y, x, 2] = q


def new():
    path = 'E:/Interleaved FLIM'
    tau_name = 'tau_NoAvr.tif'
    intensity_name = 'Intensity_NoAvr.tif'

    tau = tif.imread(path + '/' + tau_name)
    intensity = tif.imread(path + '/' + intensity_name)

    blocks = (512)
    threads = (512)
    tau_gpu = cuda.to_device(tau[0])
    intensity_gpu = cuda.to_device(intensity[0])
    RGB_gpu = cuda.to_device(np.zeros((tau.shape[0], tau.shape[1], 3), dtype=np.uint8))

    flim_to_rgb[blocks, threads](RGB_gpu, tau_gpu, intensity_gpu, 0, 2, intensity.max())    

    t = time.time()
    for k in range(1000):
        flim_to_rgb[blocks, threads](RGB_gpu, tau_gpu, intensity_gpu, 0, 2, intensity.max())
    print((time.time()-t)/1000)

    RGB_cpu = np.zeros((tau.shape[1], tau.shape[2], 3), dtype=np.uint8)
    RGB_gpu.copy_to_host(RGB_cpu)

    viewer = napari.Viewer()
    viewer.add_image(RGB_cpu)
    napari.run()