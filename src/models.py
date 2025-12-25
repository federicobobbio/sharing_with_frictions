
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
This script builds the mathematical model solving Formulations 7 and 10, and the unconstrained version of Formulation 10 from the paper 
"Sharing with Frictions: Limited Transfers and Costly Inspections."
"""

import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import matplotlib.cm as cm
from matplotlib.patches import Patch

import os, itertools
from matplotlib import cm      # colour maps

from cycler import cycler





##################################################################################
# Build the unconstrained model
##################################################################################
def build_model_unconstrained(
    n_u=30,                   # u–grid points for each firm
    n_alpha=30,             # α–grid points
    alpha_bar=1.0,
    u_bar=1.0,
    v=2,
    K=0,
    t_type="power", # "indep"
    f_type="uniform", f_mu=None, f_sigma=None,
    g_type="uniform", g_mu=None, g_sigma=None
):
    """
    Decision variables (continuous in [0,1]):

        a[α, u]
        c[α, u]

    The joint grid is the Cartesian product α×u.
    All integrals are approximated by Riemann sums with step sizes da, du.

    Returns dict with model, alpha_vals, u_vals, du, da, num_firms
    """

    # --------------------------------------------------
    # 1)  Discretisation
    # --------------------------------------------------
    alpha_vals = np.linspace(0, alpha_bar, n_alpha + 1)
    u_vals     = np.linspace(0, u_bar, n_u + 1)
    da = alpha_vals[1] - alpha_vals[0]
    du = u_vals[1]     - u_vals[0]

    # densities -------------------------------------------------------------
    if f_type.lower() == "uniform":
        f_alpha = lambda a: 1/alpha_bar
    else:
        if f_mu is None:     f_mu    = alpha_bar/2
        if f_sigma is None:  f_sigma = alpha_bar/4
        a_std = (0-f_mu)/f_sigma; b_std = (alpha_bar-f_mu)/f_sigma
        f_alpha = lambda a: truncnorm.pdf(a, a_std, b_std,
                                          loc=f_mu, scale=f_sigma)

    if g_type.lower() == "uniform":
        g_u = lambda a: 1/u_bar
    else:
        if g_mu is None:    g_mu    = u_bar/2
        if g_sigma is None: g_sigma = u_bar/4
        a_std_u = (0-g_mu)/g_sigma; b_std_u = (u_bar-g_mu)/g_sigma
        g_u = lambda u: truncnorm.pdf(u, a_std_u, b_std_u,
                                          loc=g_mu, scale=g_sigma)


    # interference ----------------------------------------------------------
    def t(u, alpha):
        if t_type == "power":
            return alpha*u
        elif t_type == "indep":
            return alpha
        elif t_type == "exp":
            return alpha_bar*u_bar*(1-np.exp(-alpha*u))
        elif t_type == "logit":
            return alpha_bar*u_bar*((alpha*u)/(1+alpha*u))
        else:
            raise ValueError("unsupported t_type")

    # --------------------------------------------------
    # 2)  Build Gurobi model
    # --------------------------------------------------
    m = gp.Model("unconstrained")
    m.setParam("OutputFlag", 0)
    m.setParam("FeasibilityTol", 1e-8)
    m.setParam("OptimalityTol",   1e-8)

    # attach the density functions so we can read them later  ★ NEW
    m._f_alpha = f_alpha          # 1-D α–pdf
    m._g_u     = g_u              # list of 1-D u-pdfs


    # index helpers ---------------------------------------------------------
    grid     = list(itertools.product(alpha_vals, u_vals))

    # variables -------------------------------------------------------------
    a_var = m.addVars(grid,     lb=0, ub=1, name="a_I")
    c_var   = m.addVars(grid,     lb=0, ub=1, name="c")

    # --------------------------------------------------
    # 3)  Objective  (Riemann sum)
    # --------------------------------------------------
    obj = gp.LinExpr()

    for (a, u) in grid:
        fa      = f_alpha(a)                       # f(α)
        gu = g_u(u)                                # g(u)
        weight  = fa * gu * da * du 

        if t(u, a)>=v: # interference > v
            obj += (v -u)* a_var[a, u] * weight
            
        else:         # not big interference
            surplus = t(u, a) - u   
            obj += surplus * a_var[a, u] * weight            

    m.setObjective(obj, GRB.MAXIMIZE)
            
    m.update()
    m.optimize()

    return dict(
        model      = m,
        alpha_vals = alpha_vals,
        u_vals     = u_vals,
        da         = da,
        du         = du,
        grid       = grid,
        a_I        = a_var,
        f_alpha    = f_alpha,     
        g_u        = g_u          
    )





##################################################################################
# Build and solve the reduced model with the inequality at the budget constraint
##################################################################################
def build_model(
    n_u=30,                   # u–grid points
    n_alpha=30,               # α–grid points
    alpha_bar=1.0,
    u_bar=1.0,
    v=2,
    K=0,
    t_type="power",           # interference type = "indep" or "power"
    f_type="uniform", f_mu=None, f_sigma=None,
    g_type="uniform", g_mu=None, g_sigma=None
):
    """
    Decision variables (continuous in [0,1]):

        a[α, u]
        c[α, u]

    The joint grid is the Cartesian product α×u.
    All integrals are approximated by Riemann sums with step sizes da, du.

    Returns dict with model, alpha_vals, u_vals, du, da, num_firms
    """

    # --------------------------------------------------
    # 1)  Discretisation
    # --------------------------------------------------
    alpha_vals = np.linspace(0, alpha_bar, n_alpha + 1)
    u_vals     = np.linspace(0, u_bar, n_u + 1)
    da = alpha_vals[1] - alpha_vals[0]
    du = u_vals[1]     - u_vals[0]

    # densities -------------------------------------------------------------
    if f_type.lower() == "uniform":
        f_alpha = lambda a: 1/alpha_bar
    else:
        if f_mu is None:     f_mu    = alpha_bar/2
        if f_sigma is None:  f_sigma = alpha_bar/4
        a_std = (0-f_mu)/f_sigma; b_std = (alpha_bar-f_mu)/f_sigma
        f_alpha = lambda a: truncnorm.pdf(a, a_std, b_std,
                                          loc=f_mu, scale=f_sigma)

    if g_type.lower() == "uniform":
        g_u = lambda a: 1/u_bar
    else:
        if g_mu is None:    g_mu    = u_bar/2
        if g_sigma is None: g_sigma = u_bar/4
        a_std_u = (0-g_mu)/g_sigma; b_std_u = (u_bar-g_mu)/g_sigma
        g_u = lambda u: truncnorm.pdf(u, a_std_u, b_std_u,
                                          loc=g_mu, scale=g_sigma)


    # interference ----------------------------------------------------------
    def t(u, alpha):
        if t_type == "power":
            return alpha*u
        elif t_type == "indep":
            return alpha
        elif t_type == "exp":
            return alpha_bar*u_bar*(1-np.exp(-alpha*u))
        elif t_type == "logit":
            return alpha_bar*u_bar*((alpha*u)/(1+alpha*u))
        else:
            raise ValueError("unsupported t_type")

    # --------------------------------------------------
    # 2)  Build Gurobi model
    # --------------------------------------------------
    m = gp.Model("constrained")
    m.setParam("OutputFlag", 0)
    m.setParam("FeasibilityTol", 1e-8)
    m.setParam("OptimalityTol",   1e-8)

    # attach the density functions so we can read them later  ★ NEW
    m._f_alpha = f_alpha          # 1-D α–pdf
    m._g_u     = g_u              # list of 1-D u-pdfs


    # index helpers ---------------------------------------------------------
    grid     = list(itertools.product(alpha_vals, u_vals))

    # variables -------------------------------------------------------------
    a_var = m.addVars(grid,     lb=0, ub=1, name="a_I")
    c_var   = m.addVars(grid,     lb=0, ub=1, name="c")

    # --------------------------------------------------
    # 3)  Objective  (Riemann sum)
    # --------------------------------------------------
    obj = gp.LinExpr()

    for (a, u) in grid:
        fa      = f_alpha(a)                       # f(α)
        gu = g_u(u)                                # g(u)
        weight  = fa * gu * da * du 

        if t(u, a)>=v: # interference > v
            obj += (v -u)* a_var[a, u] * weight
            
        else:         # not big interference
            surplus = t(u, a) - u   
            obj += surplus * a_var[a, u] * weight            

    m.setObjective(obj, GRB.MAXIMIZE)

    # --------------------------------------------------
    #       Constraints
    #     – inspection budget
    #     – incentive compatibility for the incumbent
    #     – monotonicity in α and in each u_j
    #     – feasibility of inspection vs allocation
    # --------------------------------------------------
  

    # monotone in α [it is necessary from the monotonicity in α>0]
    for k in range(len(alpha_vals)-1):
        a_low  = alpha_vals[k];   a_up = alpha_vals[k+1]
        for u in u_vals:
            m.addConstr(a_var[a_low, u] <= a_var[a_up, u])
    
    # monotonicity in α>0: a(0,u)<=a(α,u) for every α>0
    #for k in range(1,len(alpha_vals)):
    #    for u in u_vals:
    #        m.addConstr(a_var[alpha_vals[0], u] <= a_var[alpha_vals[k], u])
  

    # monotone in u
    for idx_u in range(len(u_vals)-1):
        u_low  = u_vals[idx_u];   u_up = u_vals[idx_u+1]
        for a in alpha_vals:
            m.addConstr(a_var[a, u_up] <= a_var[a, u_low])

    # inspection c ≤ a_I
    for idx in grid:
        m.addConstr(c_var[idx] <= a_var[idx])

    # --------------------------------------------------
    # IC for incumbent :  a_I(a,u)+a_I(a,u)c(a,u) ≤ a_I(0,u)
    # --------------------------------------------------
    alpha0 = alpha_vals[0]                                      
    for (a, u) in grid:                                     
        m.addConstr(                                            
            a_var[a, u] - a_var[a, u] * c_var[a, u]         
            - a_var[alpha0, u]                                
            <= 0, name=f"ICright_{a}_{u}"                   
        )                                                       

    # --------------------------------------------------
    # Budget :   Σ_j  ∫ g(u) [u_j a_j − ∫_0^{u_j} a_j] ≥ K ∫ c
    # --------------------------------------------------
    #  Left-hand side
    budget = gp.LinExpr()                                       
    #  quick index to avoid list.index inside loops
    u_to_idx = {u_vals[i]: i for i in range(len(u_vals))}       

    for (a, u) in grid:                                 
                                   
        weight  = f_alpha(a) * g_u(u) * da * du

        # cumulative ∫_0^{z} a  via grid sums
        cum_int = gp.LinExpr()                              
        idx_u   = u_to_idx[u]                             
        for z_idx in range(idx_u + 1):                          
            z = u_vals[z_idx]   
            cum_int += a_var[a,z]            
        budget += weight * (-u * a_var[a, u]          
                            + cum_int * du)                 

    #  Right-hand side
    rhs = gp.LinExpr()                                          
    for (a, u) in grid:                                     
        weight  = f_alpha(a) * g_u(u) * da * du                 
        rhs    += weight * c_var[a, u]                          
    rhs *= K                                                    

    m.addConstr(budget == rhs,  name="budget")

    m.update()
    m.optimize()

    return dict(
        model      = m,
        alpha_vals = alpha_vals,
        u_vals     = u_vals,
        da         = da,
        du         = du,
        grid       = grid,
        a_I        = a_var,
        c_var=c_var,
        f_alpha    = f_alpha,     
        g_u        = g_u          
    )





##################################################################################
# Build and solve the original model with the equality at the budget constraint
##################################################################################
def build_model_budget_equality(
    n_u=30,                   # u–grid points
    n_alpha=30,               # α–grid points
    alpha_bar=1.0,
    u_bar=1.0,
    v=2,
    K=0,
    t_type="indep",           # interference type = "indep" or "power"
    f_type="uniform", f_mu=None, f_sigma=None,
    g_type="uniform", g_mu=None, g_sigma=None
):
    """
    Decision variables (continuous in [0,1]):

        a[α, u]
        c[α, u]

    The joint grid is the Cartesian product α×u.
    All integrals are approximated by Riemann sums with step sizes da, du.

    Returns dict with model, alpha_vals, u_vals, du, da, num_firms
    """

    # --------------------------------------------------
    # 1)  Discretisation
    # --------------------------------------------------
    alpha_vals = np.linspace(0, alpha_bar, n_alpha + 1)
    u_vals     = np.linspace(0, u_bar, n_u + 1)
    da = alpha_vals[1] - alpha_vals[0]
    du = u_vals[1]     - u_vals[0]

    # densities -------------------------------------------------------------
    if f_type.lower() == "uniform":
        f_alpha = lambda a: 1/alpha_bar
    else:
        if f_mu is None:     f_mu    = alpha_bar/2
        if f_sigma is None:  f_sigma = alpha_bar/4
        a_std = (0-f_mu)/f_sigma; b_std = (alpha_bar-f_mu)/f_sigma
        f_alpha = lambda a: truncnorm.pdf(a, a_std, b_std,
                                          loc=f_mu, scale=f_sigma)

    if g_type.lower() == "uniform":
        g_u = lambda a: 1/u_bar
    else:
        if g_mu is None:    g_mu    = u_bar/2
        if g_sigma is None: g_sigma = u_bar/4
        a_std_u = (0-g_mu)/g_sigma; b_std_u = (u_bar-g_mu)/g_sigma
        g_u = lambda u: truncnorm.pdf(u, a_std_u, b_std_u,
                                          loc=g_mu, scale=g_sigma)


    # interference ----------------------------------------------------------
    def t(u, alpha):
        if t_type == "power":
            return (alpha)*(u**3)
        elif t_type == "indep":
            return (alpha)
        elif t_type == "exp":
            return alpha_bar*u_bar*(1-np.exp(-alpha*u))
        elif t_type == "logit":
            return alpha_bar*u_bar*((alpha*u)/(1+alpha*u))
        else:
            raise ValueError("unsupported t_type")

    # --------------------------------------------------
    # 2)  Build Gurobi model
    # --------------------------------------------------
    m = gp.Model("constrained")
    m.setParam("OutputFlag", 0)
    m.setParam("FeasibilityTol", 1e-8)
    m.setParam("OptimalityTol",   1e-8)

    # attach the density functions so we can read them later  ★ NEW
    m._f_alpha = f_alpha          # 1-D α–pdf
    m._g_u     = g_u              # list of 1-D u-pdfs


    # index helpers ---------------------------------------------------------
    grid     = list(itertools.product(alpha_vals, u_vals))

    # variables -------------------------------------------------------------
    a_var = m.addVars(grid,     lb=0, ub=1, name="a_I")
    c_var   = m.addVars(grid,     lb=0, ub=1, name="c")

    # --------------------------------------------------
    # 3)  Objective  (Riemann sum)
    # --------------------------------------------------
    obj = gp.LinExpr()

    for (a, u) in grid:
        fa      = f_alpha(a)                       # f(α)
        gu = g_u(u)                                # g(u)
        weight  = fa * gu * da * du 

        if t(u, a)>=v: # interference > v
            obj += (v -u)* a_var[a, u] * weight
            
        else:         # not big interference
            surplus = t(u, a) - u   
            obj += surplus * a_var[a, u] * weight            

    m.setObjective(obj, GRB.MAXIMIZE)

    # --------------------------------------------------
    #       Constraints
    #     – inspection budget
    #     – incentive compatibility for the incumbent
    #     – monotonicity in α and in each u_j
    #     – feasibility of inspection vs allocation
    # --------------------------------------------------
  

    # monotone in α [it is necessary from the monotonicity in α>0]
    for k in range(len(alpha_vals)-1):
        a_low  = alpha_vals[k];   a_up = alpha_vals[k+1]
        for u in u_vals:
            m.addConstr(a_var[a_low, u] <= a_var[a_up, u])
    
    # monotonicity in α>0: a(0,u)<=a(α,u) for every α>0
    #for k in range(1,len(alpha_vals)):
    #    for u in u_vals:
    #        m.addConstr(a_var[alpha_vals[0], u] <= a_var[alpha_vals[k], u])
  

    # monotone in u
    for idx_u in range(len(u_vals)-1):
        u_low  = u_vals[idx_u];   u_up = u_vals[idx_u+1]
        for a in alpha_vals:
            m.addConstr(a_var[a, u_up] <= a_var[a, u_low])

    # inspection c ≤ a_I
    for idx in grid:
        m.addConstr(c_var[idx] <= a_var[idx])

    # --------------------------------------------------
    # IC for incumbent :  a_I(a,u)+a_I(a,u)c(a,u) ≤ a_I(0,u)
    # --------------------------------------------------
    alpha0 = alpha_vals[0]                                      
    for (a, u) in grid:                                     
        m.addConstr(                                            
            a_var[a, u] - a_var[a, u] * c_var[a, u]         
            - a_var[alpha0, u]                                
            <= 0, name=f"ICright_{a}_{u}"                   
        )                                                       

    # --------------------------------------------------
    # Budget :   Σ_j  ∫ g(u) [u_j a_j − ∫_0^{u_j} a_j] ≥ K ∫ c
    # --------------------------------------------------
    #  Left-hand side
    budget = gp.LinExpr()                                       
    #  quick index to avoid list.index inside loops
    u_to_idx = {u_vals[i]: i for i in range(len(u_vals))}       

    for (a, u) in grid:                                 
                                   
        weight  = f_alpha(a) * g_u(u) * da * du

        # cumulative ∫_0^{z} a  via grid sums
        cum_int = gp.LinExpr()                              
        idx_u   = u_to_idx[u]                             
        for z_idx in range(idx_u + 1):                          
            z = u_vals[z_idx]   
            cum_int += a_var[a,z]            
        budget += weight * (-u * a_var[a, u]          
                            + cum_int * du)                 

    #  Right-hand side
    rhs = gp.LinExpr()                                          
    for (a, u) in grid:                                     
        weight  = f_alpha(a) * g_u(u) * da * du                 
        rhs    += weight * c_var[a, u]                          
    rhs *= K                                                    

    m.addConstr(budget == rhs,  name="budget")

    m.update()
    m.optimize()

    return dict(
        model      = m,
        alpha_vals = alpha_vals,
        u_vals     = u_vals,
        da         = da,
        du         = du,
        grid       = grid,
        a_I        = a_var,
        c_var=c_var,
        f_alpha    = f_alpha,     
        g_u        = g_u          
    )







