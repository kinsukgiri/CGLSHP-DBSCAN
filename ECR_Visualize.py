#!/usr/bin/env python3
"""Visualize the ECR diagnostic CSV files emitted by the modified DSS C program.
Usage: python3 ECR_Visualize.py <dataset_stem>
Example: python3 ECR_Visualize.py Agg
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle


def knee_index(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 0
    x1, y1 = 0.0, values[0]
    x2, y2 = float(len(values) - 1), values[-1]
    den = np.hypot(y1 - y2, x2 - x1)
    if den < 1e-10:
        return len(values) // 2
    x = np.arange(len(values), dtype=float)
    dist = np.abs((y1-y2)*x + (x2-x1)*values + x1*y2 - x2*y1) / den
    return int(np.argmax(dist))


def load(stem, suffix):
    path = Path(f"{stem}_ECR_{suffix}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the modified DSS program first.")
    return pd.read_csv(path)


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 ECR_Visualize.py <dataset_stem>")
        raise SystemExit(2)
    stem = sys.argv[1]
    pts = load(stem, "points")
    hull = load(stem, "hull")
    tris = load(stem, "triangles")
    circles = load(stem, "circles")
    radii = load(stem, "radii")
    minpts = load(stem, "minpts")
    xy = pts[["pc1", "pc2"]].to_numpy()
    eps = float(radii.iloc[knee_index(radii.radius)].radius) if len(radii) else float("nan")
    ki = knee_index(radii.radius) if len(radii) else 0

    out_pdf = Path(f"{stem}_ECR_Visualization.pdf")
    with PdfPages(out_pdf) as pdf:
        # 1: PCA projection / input point cloud
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=16, alpha=.75)
        ax.set(title="1. ECR input: normalized data projected to PCA-2D",
               xlabel="Principal component 1", ylabel="Principal component 2")
        ax.grid(alpha=.2); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 2: Convex hull
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=14, alpha=.55, label="Projected points")
        if len(hull) >= 2:
            hp = hull[["pc1", "pc2"]].to_numpy()
            hp = np.vstack([hp, hp[0]])
            ax.plot(hp[:, 0], hp[:, 1], linewidth=2.2, label="Convex hull boundary")
            ax.scatter(hp[:-1, 0], hp[:-1, 1], s=32, label="Hull vertices")
        ax.set(title="2. Convex hull: outer boundary of the projected points",
               xlabel="PC1", ylabel="PC2")
        ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3: Delaunay triangulation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=14, zorder=3)
        for row in tris.itertuples(index=False):
            ids = [int(row.a), int(row.b), int(row.c), int(row.a)]
            ax.plot(xy[ids, 0], xy[ids, 1], linewidth=.65, alpha=.65)
        if len(hull) >= 2:
            hp = hull[["pc1", "pc2"]].to_numpy(); hp = np.vstack([hp, hp[0]])
            ax.plot(hp[:, 0], hp[:, 1], linewidth=2.0, label="Convex hull")
        ax.set(title=f"3. Delaunay triangulation ({len(tris)} retained triangles)",
               xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4: Circumcircles
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(xy[:, 0], xy[:, 1], s=12, zorder=3)
        # Draw a representative subset if there are many circles, keeping the plot legible.
        stride = max(1, len(circles)//100)
        for row in circles.iloc[::stride].itertuples(index=False):
            ax.add_patch(Circle((row.cx, row.cy), row.radius, fill=False,
                                linewidth=.7, alpha=.28))
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(title=f"4. Delaunay circumcircles (showing every {stride} circle)",
               xlabel="PC1", ylabel="PC2")
        ax.grid(alpha=.2); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 5: Radius curve / knee / epsilon
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = np.arange(len(radii))
        ax.plot(x, radii.radius, linewidth=1.8, label="Sorted circumradii")
        if len(radii):
            ax.scatter([ki], [eps], s=65, zorder=4, label=f"Knee / epsilon = {eps:.6g}")
            ax.axhline(eps, linestyle="--", linewidth=1.2)
        ax.set(title="5. Sorted circumradii and knee-based epsilon selection",
               xlabel="Sorted radius index", ylabel="Circumradius")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 6: MinPts search
        fig, ax = plt.subplots(figsize=(8, 5.5))
        valid = minpts[minpts.valid == 1]
        ax.plot(minpts.minpts, minpts.silhouette, marker=".", alpha=.6,
                label="Silhouette score (all candidates)")
        if len(valid):
            ax.scatter(valid.minpts, valid.silhouette, s=38, label="Valid candidates")
        selected = minpts[minpts.selected == 1]
        if len(selected):
            sr = selected.iloc[0]
            ax.scatter([sr.minpts], [sr.silhouette], marker="*", s=180,
                       label=f"Selected MinPts = {int(sr.minpts)}")
        ax.set(title="6. MinPts selection using silhouette score",
               xlabel="MinPts candidate", ylabel="Silhouette score")
        ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    # Also export individual PNGs from the PDF stages by rerunning figures is unnecessary;
    # the multi-page PDF keeps all ECR stages together in their processing order.
    selected = minpts[minpts.selected == 1]
    chosen_mp = int(selected.iloc[0].minpts) if len(selected) else 5
    print(f"Created: {out_pdf}")
    print(f"ECR epsilon (PCA-2D radius knee): {eps:.10g}")
    print(f"Selected MinPts: {chosen_mp}")
    print("Note: this implementation does not inverse-transform PCA coordinates. It keeps the original normalized data unchanged and uses it for the later LSH/DBSCAN stage.")


if __name__ == "__main__":
    main()
