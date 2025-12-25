#****************************************************************************//
# Copyright 2025 Federico Bobbio, Randall Berry, Michael Honig, Thanh Nguyen, 
# Vijay Subramanian, Rakesh Vohra.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the conditiosn specified in the LICENSE file
#
#****************************************************************************//


"""
This script creates the plots used for Figure 1 and Figure 2.
Given the type of the interference, t_type, the plots are saved in the folder:
    results/plots/{t_type}
"""

import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import matplotlib.cm as cm
from matplotlib.patches import Patch

import os
from matplotlib import cm      # colour maps

from cycler import cycler

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap, Normalize

from matplotlib.ticker import FuncFormatter # to remove the some labels from the axis


from models import build_model

plt.rcParams.update({"font.size": 11})




def get_a_and_c(res):
    """
    Convert the solved ex-post model into two (N_alpha+1) × (n+1) numpy arrays
    that can be plotted with imshow().
    """
    alpha_vals = res["alpha_vals"]
    u_vals     = res["u_vals"]
    a_I_var    = res["a_I"]

    m_alpha = len(alpha_vals)
    m_u     = len(u_vals)
    AI  = np.zeros((m_alpha, m_u))
    CC  = np.zeros_like(AI)

    for ia, a in enumerate(alpha_vals):
        for iu, u in enumerate(u_vals):
            AI[ia, iu] = a_I_var[a, u].X
            CC[ia, iu] = res["model"].getVarByName(f"c[{a},{u}]").X
            # if you kept the variable handle:  CC[ia,iu] = c_var[a,(u,)].X
    return AI.T, CC.T



def plot_heat_for_single_K(res, save_dir, tag="", threshold=0.8, v=None, add_guides=True):
    AA, CC = get_a_and_c(res)              
    alpha_vals = np.asarray(res["alpha_vals"])
    u_vals     = np.asarray(res["u_vals"])
    extent = [alpha_vals[0], alpha_vals[-1], u_vals[0], u_vals[-1]]

    # define reference colormaps (to match overlay_K)
    cmap_a = ListedColormap(["#c8c4d6", "#fffccf"])  # greyish, pale yellow
    cmap_b = plt.cm.YlGnBu.reversed()   # light yellow to blue
    cmap_c = plt.cm.inferno             # or pick the same as overlay_C if you have it
    cmap_d = ListedColormap(["#1d1128", "#fffccf"])  # dark (near black/purple), pale yellow

    def highest_u_where(col):
        mask = col > threshold
        if np.any(mask):
            idx = np.where(mask)[0].max()
            return u_vals[idx]
        return None

    u_bot = highest_u_where(AA[:, 0])      
    u_top = highest_u_where(AA[:, -1])     

    def add_horizontal_guides(ax):
        """Draw dashed lines with labels on the left side."""
        x_left = alpha_vals[0]
        y_min, y_max = u_vals[0], u_vals[-1]
        LABEL_FSIZE = 10  # pt

        def hline(y, label):
            ax.axhline(y, linestyle='--', linewidth=1.8, color='tab:blue')
            ax.annotate(label, xy=(x_left, y),
                        xytext=(-6, 0), textcoords='offset points',  # shift a bit left
                        ha='right', va='center', fontsize=11,
                        bbox=dict(boxstyle='round,pad=0.2',
                                fc='white', ec='none', alpha=0.8))

        if u_bot is not None:
            hline(u_bot, r'$u_{\mathrm{bot}}$')
        if u_top is not None:
            hline(u_top, r'$u_{\mathrm{top}}$')
        if v is not None and (y_min <= v <= y_max):
            hline(v, r'$v$')

    # we define pastel-red
    pastel_red = np.array([1.0, 0.70, 0.70, 1.0])
    # ---------- a(α,u) heatmap ----------
    fig, ax = plt.subplots(figsize=(7,5))
    norm = Normalize(vmin=0, vmax=1)
    rgba = cmap_a(norm(AA))

    if u_bot is not None:
        idx_botp1 = np.searchsorted(u_vals, u_bot, side='right')

        mask = np.zeros_like(AA, dtype=bool)
        gt = AA > threshold  # threshold condition

        for j in range(AA.shape[1]):               # per α column
            if np.any(gt[:, j]):
                idx_step = np.where(gt[:, j])[0][0]  # first row above the step
                start = max(idx_botp1+1, idx_step)     # ensure strictly above u_bot
                mask[start:, j] = gt[start:, j]

        pastel_red = np.array([1.0, 0.70, 0.70, 1.0])
        rgba[mask] = pastel_red

    # we want to remove the label "6" from the vertical axis
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '' if np.isclose(y,6.0) else f'{int(y)}' if y.is_integer() else f'{y:g}'))


    ax.imshow(rgba, origin="lower", aspect="auto", extent=extent)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$u$", rotation=0)
    ax.yaxis.set_label_coords(-0.06, 0.9)  # push higher the label
    if add_guides:
        add_horizontal_guides(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"a_heat{tag}.png"), dpi=300)
    plt.close(fig)
    
    # ---------- c(α,u) heatmap ----------
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(CC, origin="lower", aspect="auto", extent=extent,
                   cmap=cmap_a, vmin=0, vmax=1)   # <- reuse same cmap so it's consistent
    #plt.colorbar(im, ax=ax, label=r"$c(\alpha,u)$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$u$")
    #ax.set_title(r"$c$ heat-map")
    if add_guides:
        add_horizontal_guides(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"c_heat{tag}.png"), dpi=300)
    plt.close(fig)






def generate_heatmaps(params, K=20.0, t_type="power", v=None, outdir=None, threshold=0.9, tag=None, add_guides=True):
    # -------- build model with chosen parameters --------
    parss = params.copy()
    parss["K"] = K  # override if passed separately
    parss["t_type"] = t_type  # override if passed separately
    if v is not None:
        parss["v"] = v

    resK = build_model(**parss)

    # -------- output folder --------
    if outdir is not None:
        save_dir = outdir
    else:
        save_dir = os.path.join(os.path.dirname(__file__), "..", "results", "plots", t_type)
        save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # -------- tag for filenames --------
    tag = tag if tag is not None else f"_K{int(K) if float(K).is_integer() else K}"

    # -------- generate plots --------
    plot_heat_for_single_K(
        resK,
        save_dir,
        tag=tag,
        threshold=threshold,
        v=parss["v"],
        add_guides=add_guides
    )





#########################################
# Main 
#########################################
if __name__ == "__main__":
    import argparse

    # -------- defaults --------
    default_pars = dict(
        n_u        = 100, # 500 # granularity of the discretization for u 
        n_alpha    = 100, # 500 # granularity of the discretization for alpha
        u_bar      = 10.0,  # 10.0
        alpha_bar  = 10.0,   # 10.0
        v          = 6.0,    # 6 
        K          = 20.0,    # 20          # will be overwritten by --K
        t_type     = "power",          # "power" | "indep" | "exp" | "logit"
        f_type     = "gaussian",
        f_mu       = 3.0, f_sigma = 11.5,   # 3 , 11.5
        g_type     = "gaussian",
        g_mu       = 7.0, g_sigma = 11.5   # 3 , 11.5
    )

    parser = argparse.ArgumentParser(description="Plot a(α,u) and c(α,u) heatmaps for a single K.")
    parser.add_argument("--K", type=float, default=default_pars["K"], help="Inspection cost K")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Threshold for u_bot / u_top on a(α,u)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag appended to filenames (default: _K{K})")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: plots/power or plots/indep)")
    parser.add_argument("--no-guides", action="store_true",
                        help="Disable horizontal guide lines")
    parser.add_argument("--t_type", type=str, default="power", choices=["power", "indep"])
    parser.add_argument("--v", type=float, default=None)

    args = parser.parse_args()

    # -------- build model at chosen K --------
    pars = default_pars.copy()
    pars["K"] = default_pars["K"]

    # -------- tag for filenames --------
    tag = args.tag if args.tag is not None else f"_K{int(args.K) if float(args.K).is_integer() else args.K}"


    generate_heatmaps(
        params=default_pars,
        K=default_pars["K"],
        t_type=args.t_type,
        v=args.v,
        outdir=args.outdir,
        threshold=args.threshold,
        tag=args.tag,
        add_guides=not args.no_guides
    )