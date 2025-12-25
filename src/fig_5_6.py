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
This script creates the plots used for Figures 5 and 6.
Given the type of the interference, t_type, the plots are saved in the folder:
    results/plots/{t_type}
"""



import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import os
from models import build_model, build_model_unconstrained


plt.rcParams.update({"font.size": 11})



def compute_obj_u(params, unconstrained=False):
    """
    Run the ex-post model (constrained or unconstrained) using parameters from `params`
    and return the per-u objective vector.

    Args:
        params        : dictionary of all model parameters (must include v, alpha_bar, u_bar, etc.)
        unconstrained : if True, use the unconstrained model (i.e., ignore K)

    Returns:
        u_vals : array of u values
        obj_u  : array of expected objective values per u
    """
    func = build_model_unconstrained if unconstrained else build_model
    args = params.copy()

    if unconstrained:
        args.pop("K", None)  # ensure K isn't passed to the unconstrained model

    res = func(**args)

    a_I = res['a_I']
    alpha_vals = res['alpha_vals']
    u_vals     = res['u_vals']
    da         = res['da']
    du         = res['du']
    f_alpha    = res['f_alpha']
    g_u        = res['g_u']

    # Extract needed scalars
    v_val = params["v"]
    alpha_bar = params["alpha_bar"]
    u_bar = params["u_bar"]
    t_type = params["t_type"]

    # Define interference function
    def t_val(a, u):
        if t_type == 'power':
            return a * u
        elif t_type == 'indep':
            return a
        elif t_type == 'exp':
            return alpha_bar * u_bar * (1 - np.exp(-a * u))
        elif t_type == 'logit':
            return alpha_bar * u_bar * ((a * u) / (1 + a * u))
        else:
            raise ValueError(f"unsupported t_type '{t_type}'")

    # Compute expected objective per u
    obj_u = np.zeros_like(u_vals)
    for j, u in enumerate(u_vals):
        total = 0.0
        for a in alpha_vals:
            alloc = a_I[a, u].X
            weight = f_alpha(a) * g_u(u) * da * du
            tv = t_val(a, u)
            payoff = (v_val - u) * alloc if tv > v_val else (tv - u) * alloc
            total += payoff * weight
        obj_u[j] = total

    return u_vals, obj_u


def plot_inefficiency_gap(params,
                          parameter="K",
                          values=[20, 40, 80],
                          t_type="indep"):
    """
    Sweep over a parameter (K or v), compute inefficiency gap curves, and plot them.

    Args:
        params    : dictionary of model parameters (includes K, v, etc.)
        parameter : "K" or "v" — the parameter to sweep
        values    : list of values to sweep over
        t_type    : type of the interference
    """

    params = params.copy()
    params["t_type"] = t_type

    # Start plot
    plt.figure(figsize=(12, 8))

    for val in values:
        current_params = params.copy()
        current_params[parameter] = val

        # Compute constrained model (second best solution sb_u)
        u_vals, sb_u = compute_obj_u(current_params, unconstrained=False)

        # Compute un-constrained model (first best solution sb_u)
        u_vals, fb_u = compute_obj_u(current_params, unconstrained=True)

        # Compute gap
        with np.errstate(divide='ignore', invalid='ignore'):
            gap_u = (fb_u - sb_u) / fb_u
        label = f"{parameter}={val}"
        plt.plot(u_vals, gap_u, linestyle='--', linewidth=2, label=label)

    # Labels and formatting
    plt.xlabel(r"$u$", fontsize=11)
    plt.ylabel("inefficiency gap", fontsize=11)
    plt.yscale("log")
    plt.legend(loc="best", fontsize=11)
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    
    out_dir = os.path.join(os.path.dirname(__file__), "..","results","plots", t_type)
    os.makedirs(out_dir, exist_ok=True)
    filename = f'gap_vs_u_{parameter}_{t_type}.pdf'
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()






if __name__ == "__main__":
    import argparse

    # --- Default base parameters ---
    default_pars = dict(
        n_u        = 100,
        n_alpha    = 100,
        u_bar      = 10.0,
        alpha_bar  = 10.0,
        K          = 20.0,
        v          = 6.0,
        t_type     = "indep",  # or "power"
        f_type     = "gaussian",
        f_mu       = 5.0,
        f_sigma    = 7.0,
        g_type     = "gaussian",
        g_mu       = 5.0,
        g_sigma    = 7.0
    )

    # --- CLI interface ---
    parser = argparse.ArgumentParser(description="Plot inefficiency gap vs u by sweeping over K or v.")
    parser.add_argument("--parameter", type=str, choices=["K", "v"], default="K",
                        help="Parameter to sweep: 'K' or 'v'")
    parser.add_argument("--values", type=float, nargs="+", default=[20.0, 40.0, 80.0],
                        help="Values to sweep for the chosen parameter")
    parser.add_argument("--t_type", type=str, default="indep", choices=["indep", "power", "exp", "logit"],
                        help="Interference type")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for output filename")
    parser.add_argument("--no-save", action="store_true", help="Do not save figure")
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure")
    args = parser.parse_args()

    # --- Prepare parameters ---
    params = default_pars.copy()
    params["t_type"] = args.t_type

    # --- Run the plot ---
    plot_inefficiency_gap(params=params, parameter="K", values=[20,44], t_type="power")