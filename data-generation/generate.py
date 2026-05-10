"""
Synthetic Tumor Microenvironment Dataset
========================================
Biologically-inspired 2D binary classification dataset.

Structure
---------
C1 (minority, label=0) — Viable tumour rim
    Proliferating cells forming a ring around a necrotic core.
    Sampled from the viable band of an anisotropically-grown Eden colony.

C2 (majority, label=1) — Immune / stromal infiltrate
    Surrounding immune and stromal cells.
    Asymmetrically distributed: dense on the vascularised "hot" side,
    sparse on the immune-excluded "cold" side.

Key biological properties reproduced
-------------------------------------
1. Anisotropic colony growth  → vessel-biased Eden model
2. Necrotic core              → oxygen diffusion limit (cells too deep = dead)
3. Viable rim only for C1     → hollow ring, not a solid blob
4. Asymmetric C2 ring         → hot/cold immune phenotype
5. Class imbalance            → 1 : 4  (C1 : C2)
6. Spatially decaying C2      → density drops away from tumour boundary

Dependencies
------------
    pip install numpy pandas matplotlib scipy

Usage
-----
    python generate_tumor_dataset.py
    # Outputs: tumor_dataset.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    gaussian_filter,
)


print("Generating synthetic tumor microenvironment dataset...")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — tweak these to explore different dataset properties
# ─────────────────────────────────────────────────────────────────────────────

SEED         = 137       # random seed — change for a different colony shape
GRID         = 300       # lattice size (GRID × GRID pixels)
TARGET_CELLS = 4800      # how many lattice cells to grow in the colony

# Growth anisotropy
VESSEL_DEG   = 32        # angle of simulated blood vessel (degrees from +x)
BIAS         = 2.8       # strength of directional growth bias
                         # 0 = isotropic Eden, higher = more elongated

# Necrotic core  (oxygen diffusion limit)
NECROSIS_D   = 20        # cells deeper than this (pixels) become necrotic
VIABLE_MIN   = 2         # skip the outermost 2px fringe (noisy boundary)

# C2 immune ring
DILATION     = 38        # pixel width of the immune ring around the colony
HOT_STRENGTH = 2.2       # asymmetry strength; higher = more hot/cold contrast
DIST_DECAY   = 14.0      # exponential decay of C2 density with distance

# Dataset size
N_C1 = 200               # minority class samples (viable tumour rim)
N_C2 = 800               # majority class samples (immune infiltrate)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — ANISOTROPIC EDEN GROWTH
# ─────────────────────────────────────────────────────────────────────────────
# Classic Eden (1961): at each step, pick a random perimeter cell and
# colonise one of its empty neighbours — producing rough, fractal-like
# boundaries.
#
# Here we weight the perimeter selection by each cell's projection onto
# the vessel axis, so cells "in front of" the vessel are more likely to
# grow — producing a biologically realistic elongated colony.

print("Step 1 — Growing anisotropic tumor colony...")

rng = np.random.default_rng(SEED)

CX, CY = GRID // 2, GRID // 2          # start from centre of grid
VESSEL_ANGLE = np.deg2rad(VESSEL_DEG)
vx = np.cos(VESSEL_ANGLE)              # vessel unit vector
vy = np.sin(VESSEL_ANGLE)

grid = np.zeros((GRID, GRID), dtype=bool)
grid[CX, CY] = True

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-connectivity

def empty_neighbours(r, c):
    """Return unoccupied 4-connected neighbours within grid bounds."""
    return [
        (r + dr, c + dc)
        for dr, dc in DIRS
        if 0 < r + dr < GRID - 1
        and 0 < c + dc < GRID - 1
        and not grid[r + dr, c + dc]
    ]

perimeter = {(CX, CY)}

while grid.sum() < TARGET_CELLS:
    if not perimeter:
        break

    plist = np.array(list(perimeter))

    # Project each perimeter cell onto the vessel axis
    # Rows increase downward, so negate for y
    dy_p = -(plist[:, 0] - CX) / (GRID / 2)
    dx_p =  (plist[:, 1] - CY) / (GRID / 2)
    proj = vx * dx_p + vy * dy_p

    # Softmax-style weighting: cells further along vessel axis = more likely
    w = np.exp(BIAS * proj)
    w /= w.sum()

    # Pick a weighted-random perimeter cell and fill a random empty neighbour
    idx  = rng.choice(len(plist), p=w)
    cell = tuple(plist[idx])
    nbrs = empty_neighbours(*cell)

    if not nbrs:
        perimeter.discard(cell)
        continue

    nr, nc = nbrs[rng.integers(len(nbrs))]
    grid[nr, nc] = True

    if not empty_neighbours(*cell):
        perimeter.discard(cell)
    perimeter.add((nr, nc))

print(f"  Colony grown: {grid.sum()} lattice cells")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — NECROTIC CORE  (oxygen diffusion limit)
# ─────────────────────────────────────────────────────────────────────────────
# distance_transform_edt measures each occupied cell's distance to the
# nearest boundary. Cells too deep inside cannot receive oxygen → necrotic.
# C1 is sampled only from the viable band between VIABLE_MIN and NECROSIS_D.

print("Step 2 — Computing necrotic core and viable rim...")

dist_inward = distance_transform_edt(grid)      # depth inside colony (pixels)

necrotic_mask = grid & (dist_inward > NECROSIS_D)
viable_mask   = grid & (dist_inward > VIABLE_MIN) & (dist_inward <= NECROSIS_D)

print(f"  Viable rim : {viable_mask.sum()} pixels")
print(f"  Necrotic core: {necrotic_mask.sum()} pixels  (not sampled)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — ASYMMETRIC IMMUNE RING  (C2)
# ─────────────────────────────────────────────────────────────────────────────
# Dilate the colony outward to define the ring region.
# Within that ring, assign each pixel a sampling weight based on:
#   (a) its projection onto the vessel axis  → hot/cold asymmetry
#   (b) its distance from the colony edge    → closer = denser infiltration

print("Step 3 — Building asymmetric immune infiltrate ring...")

outer_ring = binary_dilation(grid, iterations=DILATION) & ~grid

rr, rc = np.where(outer_ring)

dy_r = -(rr - CX) / (GRID / 2)
dx_r =  (rc - CY) / (GRID / 2)

proj_r   = vx * dx_r + vy * dy_r
hot_w    = np.exp(HOT_STRENGTH * proj_r)           # hot/cold asymmetry

dist_from_tumor = distance_transform_edt(~grid)[rr, rc]
dist_w = np.exp(-dist_from_tumor / DIST_DECAY)     # proximity weighting

c2_weight  = hot_w * dist_w
c2_weight /= c2_weight.sum()                       # normalise to probabilities

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SAMPLE CONTINUOUS POINTS
# ─────────────────────────────────────────────────────────────────────────────
# Subsample from valid lattice positions and add sub-pixel jitter so the
# final dataset is continuous rather than sitting on integer coordinates.

print("Step 4 — Sampling continuous data points...")

SCALE  = 5.0 / (GRID / 2)    # lattice → continuous coordinate scale
JITTER = SCALE * 0.65         # ≈ ⅔ of one lattice cell width

def to_xy(row, col):
    """Convert lattice (row, col) to continuous (x, y) coordinates."""
    x =  (col - CY) * SCALE
    y = -(row - CX) * SCALE   # flip row so +y points up
    return x, y

# C1: uniform subsample from viable rim
vr, vc = np.where(viable_mask)
idx_c1 = rng.choice(len(vr), size=N_C1, replace=False)
c1x, c1y = to_xy(vr[idx_c1], vc[idx_c1])
c1_pts = np.column_stack([c1x, c1y]) + rng.normal(0, JITTER, (N_C1, 2))

# C2: weighted subsample from immune ring
idx_c2 = rng.choice(len(rr), size=N_C2, replace=False, p=c2_weight)
c2x, c2y = to_xy(rr[idx_c2], rc[idx_c2])
c2_pts = np.column_stack([c2x, c2y]) + rng.normal(0, JITTER, (N_C2, 2))

c1_mean = c1_pts.mean(axis=0)
c2_mean = c2_pts.mean(axis=0)
sep     = np.linalg.norm(c1_mean - c2_mean)

print(f"  C1 centroid : ({c1_mean[0]:.3f}, {c1_mean[1]:.3f})")
print(f"  C2 centroid : ({c2_mean[0]:.3f}, {c2_mean[1]:.3f})")
print(f"  Centroid separation : {sep:.3f} units")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — BUILD DATAFRAME AND SAVE
# ─────────────────────────────────────────────────────────────────────────────

X  = np.vstack([c1_pts, c2_pts])
y  = np.array([0] * N_C1 + [1] * N_C2)   # 0 = tumour rim, 1 = immune ring

df = pd.DataFrame({
    'x1':    X[:, 0],
    'x2':    X[:, 1],
    'label': y,
})

df.to_csv('tumor_dataset.csv', index=False)
print(f"\nDataset saved → tumor_dataset.csv")
print(f"  Total samples : {len(df)}")
print(f"  C1 (label=0)  : {(y==0).sum()}  ({100*(y==0).mean():.1f}%)")
print(f"  C2 (label=1)  : {(y==1).sum()}  ({100*(y==1).mean():.1f}%)")
print(f"  Imbalance     : 1 : {(y==1).sum() // (y==0).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

print("\nStep 6 — Plotting...")

BG      = '#100c14'
COL_C1  = '#d8b4fe'    # lavender  — viable tumour cells (hematoxylin)
COL_C2H = '#fb923c'    # orange    — hot immune infiltrate
COL_C2C = '#fda4af'    # pale pink — cold / excluded immune cells
COL_VES = '#38bdf8'    # sky blue  — vessel axis

fig, ax = plt.subplots(figsize=(8, 8), facecolor=BG)
ax.set_facecolor(BG)

xg = np.linspace(-5, 5, GRID)

# Background tissue heatmap
smooth_colony = gaussian_filter(grid.astype(float), sigma=4)
he_cmap = LinearSegmentedColormap.from_list(
    'he', ['#100c14', '#2d1b3d', '#4a1c6e'], N=256)
ax.imshow(smooth_colony, extent=[-5, 5, -5, 5], cmap=he_cmap,
          alpha=0.45, zorder=1, origin='lower')

# Necrotic core overlay
nec_cmap = LinearSegmentedColormap.from_list(
    'nec', ['#100c14', '#2a1810', '#4a2a18'], N=128)
ax.imshow(gaussian_filter(necrotic_mask.astype(float), sigma=3),
          extent=[-5, 5, -5, 5], cmap=nec_cmap,
          alpha=0.7, zorder=2, origin='lower')

# Colony and necrosis boundary contours
ax.contour(xg, xg, np.flipud(grid.astype(float)),
           levels=[0.5], colors=['#a855f7'], linewidths=1.8, alpha=0.8, zorder=5)
ax.contour(xg, xg, np.flipud(necrotic_mask.astype(float)),
           levels=[0.5], colors=['#92400e'], linewidths=1.1,
           linestyles='--', alpha=0.55, zorder=5)

# C2: colour by hot/cold projection
proj_c2 = vx * (c2_pts[:, 0] / 5.0) + vy * (c2_pts[:, 1] / 5.0)
t = (proj_c2 - proj_c2.min()) / (proj_c2.max() - proj_c2.min() + 1e-9)

def hex_to_rgb(h):
    return [int(h[i:i+2], 16) / 255 for i in (1, 3, 5)]

rc_cold, gc_cold, bc_cold = hex_to_rgb(COL_C2C)
rc_hot,  gc_hot,  bc_hot  = hex_to_rgb(COL_C2H)
c2_colors = np.column_stack([
    rc_cold * (1 - t) + rc_hot * t,
    gc_cold * (1 - t) + gc_hot * t,
    bc_cold * (1 - t) + bc_hot * t,
])

ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c=c2_colors, s=25, alpha=0.18,
           linewidths=0, zorder=6)
ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c=c2_colors, s=8,  alpha=0.72,
           linewidths=0, zorder=7, label=f'C2 — Immune infiltrate  (n={N_C2})')

ax.scatter(c1_pts[:, 0], c1_pts[:, 1], c=COL_C1, s=40, alpha=0.22,
           linewidths=0, zorder=8)
ax.scatter(c1_pts[:, 0], c1_pts[:, 1], c=COL_C1, s=13, alpha=0.95,
           linewidths=0, zorder=9, label=f'C1 — Viable tumour rim  (n={N_C1})')

# Centroids
ax.scatter(*c1_mean, c='white', s=90, marker='x', linewidths=2.2, zorder=11)
ax.scatter(*c2_mean, c='white', s=90, marker='+', linewidths=2.2, zorder=11)

# Vessel arrow
arr_len = 1.5
ax.annotate('', xy=(-4 + arr_len * vx, -4.3 + arr_len * vy),
            xytext=(-4, -4.3),
            arrowprops=dict(arrowstyle='->', color=COL_VES, lw=2.0,
                            mutation_scale=13))
ax.text(-4 + arr_len * vx * 0.5, -4.3 + arr_len * vy * 0.5 - 0.35,
        'vessel axis', color=COL_VES, fontsize=8.5, ha='center')

ax.text( 3.8 * vx,  3.8 * vy, 'HOT (inflamed)',  color=COL_C2H, fontsize=9,
        ha='center', bbox=dict(facecolor='#1a0800', alpha=0.6, pad=3,
                               edgecolor='none'))
ax.text(-3.2 * vx, -3.2 * vy, 'COLD (excluded)', color='#94a3b8', fontsize=9,
        ha='center', bbox=dict(facecolor='#08080f', alpha=0.6, pad=3,
                               edgecolor='none'))

legend_elems = [
    mpatches.Patch(color=COL_C1,   label=f'C1 — Viable tumour rim  (n={N_C1})'),
    mpatches.Patch(color='#78350f', label='Necrotic core  (not sampled)'),
    mpatches.Patch(color=COL_C2H,  label=f'C2 — Immune infiltrate  (n={N_C2})'),
]
ax.legend(handles=legend_elems, frameon=False, labelcolor='white',
          fontsize=9.5, loc='upper left')

ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
ax.set_aspect('equal')
ax.tick_params(colors='#554', labelsize=9)
for sp in ax.spines.values():
    sp.set_edgecolor('#332')
ax.set_title('Synthetic Tumour Microenvironment Dataset\n'
             'Biased Eden Growth  ·  Necrotic Core  ·  Asymmetric Immune Infiltration',
             color='white', fontsize=11, fontweight='bold', pad=11)
ax.set_xlabel('x₁', color='#887', fontsize=11)
ax.set_ylabel('x₂', color='#887', fontsize=11)

plt.tight_layout()
plt.savefig('tumor_dataset_plot.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
print("Plot saved → tumor_dataset_plot.png")


