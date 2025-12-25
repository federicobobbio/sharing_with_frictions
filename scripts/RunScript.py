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

import os
import sys

# Add src/ to Python path
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC_DIR)

print("Current file:", __file__)
print("Resolved SRC_DIR:", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from fig_1_2 import generate_heatmaps

from fig_3_4 import overlay_heatmaps

from fig_5_6 import plot_inefficiency_gap

from fig_7_8 import new_overlay_heatmaps

# Define shared default parameters
default_pars = dict(
    n_u        = 500,
    n_alpha    = 500,
    u_bar      = 10.0,
    alpha_bar  = 10.0,
    K          = 20.0,
    v          = 6.0,
    f_type     = "gaussian",
    f_mu       = 5.0,
    f_sigma    = 7.0,
    g_type     = "gaussian",
    g_mu       = 5.0,
    g_sigma    = 7.0
)



if __name__ == "__main__":
    print("Uncomment the line of interest to generate that figure.")

    # Figure 1. Optimal allocation and inspection heatmaps for independent interference.
    #generate_heatmaps(default_pars, K=50.0, t_type="indep", v=6.0, tag="_fig1", outdir=os.path.abspath(os.path.join(SRC_DIR, "..", "results", "plots", "indep")))

    # Figure 2. Optimal allocation and inspection heatmaps for power interference.
    #generate_heatmaps(default_pars, K=50.0, t_type="power", v=6.0, tag="_fig2", outdir=os.path.abspath(os.path.join(SRC_DIR, "..", "results", "plots", "power")))

    # Figure 3, top row. 
    #overlay_heatmaps(params=default_pars, parameter="v", values=[2, 4], save_dir=os.path.join("..", "results", "plots", "indep"), t_type="indep")

    # Figure 3, bottom row. 
    #overlay_heatmaps(params=default_pars, parameter="v", values=[2, 4], save_dir=os.path.join("..", "results", "plots", "power"), t_type="power")

    # Figure 4, top row. 
    #overlay_heatmaps(params=default_pars, parameter="K", values=[25, 50], save_dir=os.path.join("..", "results", "plots", "indep"), t_type="indep")

    # Figure 4, bottom row. 
    #overlay_heatmaps(params=default_pars, parameter="K", values=[25, 50], save_dir=os.path.join("..", "results", "plots", "power"), t_type="power")

    # Figure 5, left. 
    #plot_inefficiency_gap(params=default_pars, parameter="K", values=[25, 50], t_type="indep")

    # Figure 5, right. 
    #plot_inefficiency_gap(params=default_pars, parameter="K", values=[25, 50], t_type="power")

    # Figure 6, left. 
    #plot_inefficiency_gap(params=default_pars, parameter="v", values=[2,4], t_type="indep")

    # Figure 6, right. 
    #plot_inefficiency_gap(params=default_pars, parameter="v", values=[2,4], t_type="power")

    # For simplicity, we run the plots in the Appendix with a coarser discretization.
    default_pars["n_u"] = 100
    default_pars["n_alpha"] = 100

    # Figure 7. Small K and independent interference.
    #new_overlay_heatmaps(default_pars, parameter="K", values=[ 0.1, 0.5], save_dir=os.path.join("..", "results", "plots", "indep"), t_type='indep')

    # Figure 8. Small K and power interference.
    #new_overlay_heatmaps(default_pars, parameter="K", values=[ 0.1, 0.5], save_dir=os.path.join("..", "results", "plots", "power"), t_type='power')


    



