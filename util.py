import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.backends.backend_pdf import PdfPages
from cdecimal import Decimal
import shelve

np.random.seed(0)

#---------------------------------------------------------------------------
# variables to save and load during each run
#---------------------------------------------------------------------------

save_load_vars = ['time',
                  'temp',
                  'load',
                  'occ',
                  'obs',
                  'num_times',
                  'num_obs',
                  'num_scens',
                  'num_params',
                  'param_names',
                  'scens',
                  'prior',
                  'likelihood',
                  'posterior',
                  'pri_pred_10',
                  'pri_pred_50',
                  'pri_pred_90',
                  'post_pred_10',
                  'post_pred_50',
                  'post_pred_90']

#---------------------------------------------------------------------------
# model (load is piecewise linear function of temp)
#---------------------------------------------------------------------------

def model(const_coeff, change_points, temp_coeffs, temp):
    temp_comps = np.full((len(temp),len(change_points)+1), np.nan)
    temp_comps[:,0] = temp * (temp <= change_points[0]) + change_points[0] * (temp > change_points[0])
    for i in xrange(1,len(change_points)):
        temp_comps[:,i] = (temp-change_points[i-1]) * (temp > change_points[i-1]) * (temp <= change_points[i])
        temp_comps[:,i] += (change_points[i]-change_points[i-1]) * (temp > change_points[i])
    temp_comps[:,-1] = (temp-change_points[-1]) * (temp > change_points[-1])
    mat = np.column_stack((np.ones(len(temp)), temp_comps))
    coeffs = np.concatenate((np.array([const_coeff]), np.array(temp_coeffs)))
    load = np.dot(mat, coeffs)
    return load

#---------------------------------------------------------------------------
# probability density function of error between observations and predictions
#---------------------------------------------------------------------------

def errorPDF(obs, pred):
    # err_var = 25000.0
    # pdf_min = 0.0
    exp_arg = np.array([Decimal(i) for i in -0.5/err_var*(obs-pred)**2])
    pdf = Decimal(1.0/np.sqrt(2*err_var*np.pi)) * np.exp(exp_arg)
    pdf[pdf < pdf_min] = Decimal(pdf_min)
    return pdf

#---------------------------------------------------------------------------
# cumulative distribution function plot of values with probabilities
#---------------------------------------------------------------------------

def CDFPlot(ax, vals, probs, **kwargs):
    num_bins = 100
    bins = np.linspace(np.min(vals), np.max(vals)+1e-16, num_bins)
    heights = np.full(num_bins-1, np.nan)
    for b in xrange(num_bins-1):
        idx = (vals >= bins[b]) & (vals < bins[b+1])
        heights[b] = np.sum(probs[idx])
    heights = np.cumsum(heights)
    ax.bar(bins[:-1], heights, width=bins[1]-bins[0], alpha=0.5, **kwargs)

#---------------------------------------------------------------------------
# notes
#---------------------------------------------------------------------------

# parameters: m = 1..M
# observations: n = 1..N
# scenarios: k = 1..K

# each of the K scenarios is an instantion of the M parameters

# independent variable: temperature
# dependent variable: load

# theta_m,k : scalar, m^th parameter in k^th scenario
# Theta_k : Mx1 vector, all parameters in k^th scenario

# x_n : scalar, n^th observation of independent variable
# X : Nx1 vector, all observations of independent variable

# y_n : scalar, n^th observation of dependent variable
# Y : Nx1 vector, all observations of dependent variable

# z_n,k : scalar, model prediction of dependent variable for n^th observation of independent variable in k^th scenario

# f : function, model that computes z_n,k = f(Theta_k, x_n)

# P(Theta_k | X, Y) = P(X, Y | Theta_k) * P(Theta_k) / sum_k=1..K [ P(X, Y | Theta_k) * P(Theta_k) ]

# is P(Theta_k) = prod_m=1..M [ P(theta_m,k) ]? or is this how to enforce constraints on theta_m,k?

# P(X, Y | Theta_k) = prod_n=1..N [ P(x_n, y_n | Theta_k) ] because prediction errors are i.i.d. (even though temperature observations are autocorrelated)

# assume error between model predictions and observations is N(0,sigma). what should sigma be?

# P(x_n, y_n | Theta_k) = 1/sigma/sqrt(2*pi) * exp [ -1/2 * ((y_n - z_n,k) / sigma)^2 ]

#---------------------------------------------------------------------------
