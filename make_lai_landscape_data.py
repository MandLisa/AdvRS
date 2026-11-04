"""Simulate a small Sentinel-2 'landscape' with LAI for the practical session.

Every pixel is a PROSAIL canopy. Canopy and soil parameters vary smoothly in space
(Gaussian random fields), so neighbouring pixels are similar, like in real images.
Output: lai_s2_landscape.csv with x, y (pixel coordinates), Sentinel-2 bands, NDVI and LAI.
"""
import sys
import numpy as np
import pandas as pd
import prosail
from scipy.ndimage import gaussian_filter

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
rng = np.random.default_rng(SEED)
G = 120                     # grid of 120 x 120 pixels (20 m -> 2.4 km)
wl = np.arange(400, 2501)
S2 = {"B2": (490, 65), "B3": (560, 35), "B4": (665, 30), "B5": (705, 15), "B6": (740, 15),
      "B7": (783, 20), "B8": (842, 115), "B8A": (865, 20), "B11": (1610, 90), "B12": (2190, 180)}


def field(sigma, lo, hi, trend=0.0):
    f = gaussian_filter(rng.normal(size=(G, G)), sigma, mode="wrap")
    f = (f - f.min()) / (f.max() - f.min())
    xx = np.linspace(0, 1, G)[None, :].repeat(G, 0)
    f = (1 - trend) * f + trend * xx                     # optional west-east gradient
    f = (f - f.min()) / (f.max() - f.min())
    return lo + f * (hi - lo)


P = {
    "lai": field(5, 0.1, 6.8),
    "cab": field(9, 25, 70, trend=0.5),
    "cm": field(8, 0.004, 0.011),
    "cw": field(8, 0.008, 0.025),
    "ala": field(10, 40, 70, trend=0.4),
    "psoil": field(12, 0.05, 0.95, trend=0.7),           # soil gets drier/brighter to the east
    "n": field(10, 1.3, 1.9),
}

N = 1500
idx = rng.choice(G * G, N, replace=False)
ys, xs = np.unravel_index(idx, (G, G))
rows = []
for x, y in zip(xs, ys):
    p = {k: v[y, x] for k, v in P.items()}
    spec = prosail.run_prosail(n=p["n"], cab=p["cab"], car=p["cab"] / 4, cbrown=0.0, cw=p["cw"], cm=p["cm"],
                               lai=p["lai"], lidfa=p["ala"], hspot=0.05, tts=38, tto=3, psi=60,
                               typelidf=2, rsoil=1.0, psoil=p["psoil"], factor="SDR")
    r = {"x": int(x), "y": int(y)}
    for b, (c, w) in S2.items():
        v = spec[(wl >= c - w / 2) & (wl <= c + w / 2)].mean()
        r[b] = round(float(np.clip(v * (1 + rng.normal(0, 0.02)) + rng.normal(0, 0.003), 0.001, 1)), 4)
    # LAI as measured in the field: measurement error of about 0.2 + 7 % (simplified)
    lai_meas = p["lai"] + rng.normal(0, 0.2 + 0.07 * p["lai"])
    r["LAI"] = round(float(max(lai_meas, 0.0)), 2)
    rows.append(r)

df = pd.DataFrame(rows)
df["NDVI"] = ((df.B8 - df.B4) / (df.B8 + df.B4)).round(4)
df = df[["x", "y"] + list(S2) + ["NDVI", "LAI"]]
df.to_csv("lai_s2_landscape.csv", index=False)
print(df.describe().round(3).T[["mean", "min", "max"]])
