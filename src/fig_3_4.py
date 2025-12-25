#****************************************************************************//
# Copyright 2025 XXXXFederico Bobbio, Randall Berry, Michael Honig, Thanh Nguyen, 
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
This script creates the plots used for Figures 3 and 4.
Given the type of the interference, t_type, the plots are saved in the folder:
    results/plots/{t_type}
"""

import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm
plt.rcParams.update({"font.size": 11})

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

import argparse





def get_a_and_c(res):
    """
    Returns AI, CC shaped for imshow(): (n_u+1, n_alpha+1).
    Prefers stored var handles if present.
    """
    alpha_vals = res["alpha_vals"]
    u_vals     = res["u_vals"]
    a_I_var    = res["a_I"]
    c_var      = res.get("c_var", None)  

    m_alpha = len(alpha_vals)
    m_u     = len(u_vals)
    AI  = np.zeros((m_alpha, m_u))
    CC  = np.zeros_like(AI)

    for ia, a in enumerate(alpha_vals):
        for iu, u in enumerate(u_vals):
            AI[ia, iu] = a_I_var[a, u].X
            if c_var is not None:
                CC[ia, iu] = c_var[a, u].X
            else:
                # fallback if you didn't return c_var (less robust for floats)
                CC[ia, iu] = res["model"].getVarByName(f"c[{a},{u}]").X

    return AI.T, CC.T






def overlay_heatmaps(params, parameter, values, save_dir, t_type='power',
                     a_threshold=0.8, c_threshold=0.8):
    
    """
    Generate overlay contour plots of a(α,u) and c(α,u) for multiple values
    of a given parameter (e.g., v or K).

    Arguments:
        params      : dict of fixed parameters to use
        parameter   : name of the parameter to vary (e.g. "v", "K")
        values      : list of values for that parameter
        save_dir    : output directory for saving plots
        t_type      : interference type ("power", "indep", etc.)
        a_threshold : threshold for a(α,u) contours
        c_threshold : threshold for c(α,u) contours
    """

    # Prepare sweep results
    sweep_results = []
    for val in values:
        p = params.copy()
        p[parameter] = val
        p["t_type"] = t_type  # enforce t_type
        resK = build_model(**p)
        AA_K, CC_K = get_a_and_c(resK)
        sweep_results.append((val, {
            "t_vals": resK["alpha_vals"],
            "u_vals": resK["u_vals"],
            "a_grid": AA_K,
            "c_grid": CC_K,
        }))

    

    # build grids from first result
    first = sweep_results[0][1]
    t_vals = np.asarray(first["t_vals"])
    u_vals = np.asarray(first["u_vals"])
    T, U = np.meshgrid(t_vals, u_vals, indexing='xy')
    base = f"{parameter}_{t_type}"

    # Colorblind-friendly palette (CUD) + black for contrast
    colorblind_colors = [
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#CC79A7",  # reddish purple
    ]

    linestyles = ['-', '--', '-.', ':']
    style_specs = []
    for i in range(len(sweep_results)):
        color = colorblind_colors[i % len(colorblind_colors)]
        linestyle = linestyles[i % len(linestyles)]
        style_specs.append(dict(color=color, linestyle=linestyle, linewidth=2.0))

    # --------- A) a(α,u) contours only ----------
    fig, ax = plt.subplots(figsize=(8,6))
    legend_lines = []
    for (val, res), st in zip(sweep_results, style_specs):
        aK = res["a_grid"]
        ax.contour(T, U, aK > a_threshold, levels=[0.5],
                   colors=[st['color']],
                   linewidths=st['linewidth'],
                   linestyles=st['linestyle'])
        legend_lines.append(Line2D([0],[0], **st, label=f"{parameter}={val}"))

    ax.set_xlim(t_vals.min(), t_vals.max())
    ax.set_ylim(u_vals.min(), u_vals.max())
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$u$', rotation=0)
    ax.legend(handles=legend_lines, loc='lower right', framealpha=0.95, fontsize=11)
    

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"a_overlay_{base}.png"), dpi=300)
    plt.close(fig)

    # --------- B) c(α,u) contours only ----------
    fig, ax = plt.subplots(figsize=(8,6))
    legend_lines = []
    for (val, res), st in zip(sweep_results, style_specs):
        cK = res["c_grid"]
        ax.contour(T, U, cK > c_threshold, levels=[0.5],
                   colors=[st['color']],
                   linewidths=st['linewidth'],
                   linestyles=st['linestyle'])
        legend_lines.append(Line2D([0],[0], **st, label=f"{parameter}={val}"))

    ax.set_xlim(t_vals.min(), t_vals.max())
    ax.set_ylim(u_vals.min(), u_vals.max())
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel(r'$u$', rotation=0)
    ax.legend(handles=legend_lines, loc='lower right', framealpha=0.95, fontsize=11)
    
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"c_overlay_{base}.png"), dpi=300)
    plt.close(fig)






#########################################
# Main
#########################################
if __name__ == "__main__":

    default_pars = dict(
        n_u        = 100,
        n_alpha    = 100,
        u_bar      = 10.0,
        alpha_bar  = 11.0,
        K          = 50.0,
        v          = 6.0,
        f_type     = "gaussian",
        f_mu       = 5.0,
        f_sigma    = 7.0,
        g_type     = "gaussian",
        g_mu       = 5.0,
        g_sigma    = 7.0
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
    args = parser.parse_args()

    # -------- build model at chosen K --------
    pars = default_pars.copy()
    pars["K"] = default_pars["K"]

    resK = build_model(**pars)

    # -------- output folder --------
    if args.outdir is not None:
        save_dir = args.outdir
    else:
        save_dir = os.path.join(os.path.dirname(__file__), "..","results","plots", "power" if pars["t_type"] == "power" else "indep")
    os.makedirs(save_dir, exist_ok=True)

    # -------- tag for filenames --------
    tag = args.tag if args.tag is not None else f"_K{int(args.K) if float(args.K).is_integer() else args.K}"





    # ----- baseline run -------------
    pars = default_pars.copy()
    if pars["t_type"] == 'power':
        save_dir = os.path.join(os.path.dirname(__file__), "..","results","plots", 
                            "power") 
    else:
        save_dir = os.path.join(os.path.dirname(__file__), "..","results","plots", 
                            "indep") 
    os.makedirs(save_dir, exist_ok=True)

    # ----- sweep over v values ----------------------
    
    parameter = 'v' # 'K'
    list_of_parameter = [ 4, 8, 11 ] 

   

    # Sweep
    sweep_results = []
    for val in list_of_parameter:
        p = pars.copy(); p[parameter] = val
        resK = build_model(**p)
        AA_K, CC_K = get_a_and_c(resK)
        sweep_results.append((val, {
            "t_vals": resK["alpha_vals"],
            "u_vals": resK["u_vals"],
            "a_grid": AA_K,
            "c_grid": CC_K,
        }))

    overlay_heatmaps(sweep_results, save_dir, par_name=parameter, t_type=pars["t_type"])



    # Default parameter set
    default_pars = dict(
        n_u        = 100,
        n_alpha    = 100,
        u_bar      = 10.0,
        alpha_bar  = 11.0,
        K          = 50.0,
        v          = 6.0,
        f_type     = "gaussian",
        f_mu       = 5.0,
        f_sigma    = 7.0,
        g_type     = "gaussian",
        g_mu       = 5.0,
        g_sigma    = 7.0
    )

    parser = argparse.ArgumentParser(description="Test overlay_heatmaps() for Figure 3.")
    parser.add_argument("--t_type", type=str, default="indep", choices=["indep", "power"])
    parser.add_argument("--parameter", type=str, default="v", help="Parameter to sweep over (e.g., 'v' or 'K')")
    parser.add_argument("--values", type=float, nargs="+", default=[4, 8, 11],
                        help="Values of the parameter to sweep over")
    parser.add_argument("--outdir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--a_thresh", type=float, default=0.8, help="Threshold for a(α,u)")
    parser.add_argument("--c_thresh", type=float, default=0.8, help="Threshold for c(α,u)")
    args = parser.parse_args()

    # Output directory logic
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = os.path.join(os.path.dirname(__file__), "..", "results", "plots", args.t_type)

    # Run the function
    overlay_heatmaps(
        params=default_pars,
        parameter=args.parameter,
        values=args.values,
        save_dir=outdir,
        t_type=args.t_type,
        a_threshold=args.a_thresh,
        c_threshold=args.c_thresh
    )