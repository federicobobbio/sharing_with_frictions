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
This script creates the plots used for Figures 7 and 8.
Given the type of the interference, t_type, the plots are saved in the folder:
    results/plots/{t_type}
"""

import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
plt.rcParams.update({"font.size": 11})


from cycler import cycler

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from matplotlib.colors import ListedColormap, Normalize

from matplotlib.ticker import FuncFormatter # to remove the some labels from the axis


from models import build_model_budget_equality





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








def new_overlay_heatmaps(params, parameter, values, save_dir, t_type='power',
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
        resK = build_model_budget_equality(**p)
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
    fig.savefig(os.path.join(save_dir, f"a_overlay_equality_model_{base}.png"), dpi=300)
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
    fig.savefig(os.path.join(save_dir, f"c_overlay_equality_model_{base}.png"), dpi=300)
    plt.close(fig)







#########################################
# Main: just build @ K and plot heatmaps
#########################################
if __name__ == "__main__":
    import argparse

    # -------- defaults --------
    default_pars = dict(
        n_u        = 100,
        n_alpha    = 100,

        u_bar      = 10.0,
        alpha_bar  = 10.0,

        v          = 6.0,
        K          = 20.0,             
        t_type     = "power",          # "power" | "indep" | "exp" | "logit"
        f_type     = "gaussian",
        f_mu       = 5.0, f_sigma = 7,
        g_type     = "gaussian",
        g_mu       = 5.0, g_sigma = 7
    )


    # ----- sweep over K values ----------------------
    
    parameter = 'K' # 'v'
    list_of_parameter = [ 0.05, 0.1, 1] 


    new_overlay_heatmaps(default_pars, parameter=parameter, values=list_of_parameter, save_dir=os.path.abspath(os.path.join("..", "results", "plots", "indep")), t_type='indep',
                     a_threshold=0.9, c_threshold=0.9)