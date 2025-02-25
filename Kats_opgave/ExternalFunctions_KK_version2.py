# -*- coding: utf-8 -*-

"""
Created on Fri Feb 17th 2023
@author: Kathrine Kuszon
"""

'''
For author: Copy this into every new notebook:

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import scipy
from scipy import stats
from numpy import random
import seaborn as sns
from importlib import reload
from iminuit import Minuit

# Importing own functions written for this course
sys.path.append('../')
import ExternalFunctions_KK_version2 as kk

# Plotting parameters
colors = sns.color_palette("Paired", 15, desat=1)[::2]
colors = colors[1:]
sns.set_palette(colors)
sns.palplot(colors)
plt.style.use('seaborn-white')
plt.rcParams['font.size'] = 17
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['lines.linewidth'] = 3
plt.rcParams["mathtext.default"]= 'regular'

# Machine learning
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix 

# Bayesian fitting
import ultranest
from ultranest.plot import cornerplot
from ultranest.plot import PredictionBand

'''

# ============================================================
# LIBRARIES NECESSARY FOR THIS COLLECTION OF FUNCTIONS TO WORK
# ============================================================

import numpy as np 
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy import stats
import random

import seaborn as sns
colors = sns.color_palette("Paired", 10, desat=1)
sns.set_palette(colors)



# ==================================
# PROBABILITY DISTRUBUTION FUNCTIONS
# ==================================

def gaussian_pdf(x, mu, sigma):
    return stats.norm.pdf(x, mu, sigma)



# ==================================
# LIKELIHOOD FUNCTIONS
# ==================================

def likelihood(data, pdf, pars):
    return np.prod(pdf(data, *pars))

def log_likelihood(data, pdf, pars):
    f = np.log(pdf(data, *pars))
    return -np.sum(f)



# ==================================
# CALCULATING (UNBIASED) VARIANCE
# ==================================

def calc_variance(samples,bias=False):
    N = len(samples)
    if bias:
        return np.sum((samples-np.average(samples))**2)/N
    else:
        return np.sum((samples-np.average(samples))**2)/(N-1)


    
# ==================================
# CALCULATING CHI-SQUARE VALUE
# ================================== 

def chi2(y_expected,y_measured,y_err):
    return np.sum((y_expected-y_measured)**2/(y_err**2))    
    
    
# ==================================
# MONTE CARLO ACCEPT-REJECT
# ==================================

def acc_rej(func, args, xmin, xmax, N_points):
    
    """
    Function doing the Monte-Carlo accept reject method to produce random numbers
    
    PARAMETERS
    - func: Function the random generated numbers should follow
    - xmin, xmax: Values of where the function func is defined
    - N_points: Number of random points to generate
    
    RETURNS
    - x_accepted: 1d array like with the accepted x-values
    """
    
    'Random number'
    r = np.random
    
    'Generating random numbers within the fixed box'
    x = np.linspace(xmin, xmax, N_points*5)
    rnd_x = r.uniform(xmin, xmax, N_points*5)
    rnd_y = r.uniform(0, max(func(x,*args)), N_points*5)
    
    'Running now accept-reject'
    f_rnd_x = func(rnd_x, *args)          # Using function on the random x-values
    f_rnd_x_accepted = f_rnd_x > rnd_y    # Accept condition
    x_accepted = rnd_x[f_rnd_x_accepted]  # Collecting accepted x-values
    y_accepted = rnd_y[f_rnd_x_accepted]  # Only for plotting
    

    x_accepted = x_accepted[:N_points]
    y_accepted = y_accepted[:N_points]
    
    
    return x_accepted, x, y_accepted



def acc_rej_discrete(func, args, xmin, xmax, N_points):
    
    """
    Function doing the Monte-Carlo accept reject method to produce random numbers
    
    PARAMETERS
    - func: Function the random generated numbers should follow
    - xmin, xmax: Values of where the function func is defined
    - N_points: Number of random points to generate
    
    RETURNS
    - x_accepted: 1d array like with the accepted x-values
    """
    
    'Random number'
    r = np.random
    
    'Generating random numbers within the fixed box'
    x = r.randint(xmin, xmax, N_points*5)
    rnd_x = r.randint(xmin, xmax, N_points*5)
    rnd_y = r.uniform(0, max(func(x,*args)), N_points*5)
    
    'Running now accept-reject'
    f_rnd_x = func(rnd_x, *args)          # Using function on the random x-values
    f_rnd_x_accepted = f_rnd_x > rnd_y    # Accept condition
    x_accepted = rnd_x[f_rnd_x_accepted]  # Collecting accepted x-values
    y_accepted = rnd_y[f_rnd_x_accepted]  # Only for plotting
    
    x_accepted = x_accepted[:N_points]
    y_accepted = y_accepted[:N_points]
    
    return x_accepted, x, y_accepted


# ==================================
# LN(LIKELIHOOD) MINIMIZATION
# ==================================

def likelihood_fit(data, func, startparams, var_names):
    
    '''
    Parameters: data, fitfunction, startguesses/parameters, variable names
    Returns: Parameter names, parameter estimates, parameter estimate uncertainties, minimizer steps, likelihood value
    '''
    
    steps_taken = []
    
    def log_likelihood(pars): 
        f = np.log(func(data, *pars))
        steps_taken.append(pars)   # to plot minimizer steps
        return -np.sum(f)

    minuit_ullh = Minuit(log_likelihood, startparams, name=var_names)
    minuit_ullh.limits[var_names[0]] = (0,1)
    minuit_ullh.limits[var_names[1]] = (0,1)

    minuit_ullh.errordef = 0.5
    minuit_ullh.migrad()
    LLH_val = minuit_ullh.fval
    
    par = minuit_ullh.values[:]
    par_err = minuit_ullh.errors[:] 
    par_name = minuit_ullh.parameters[:]
    return par_name, par, par_err, steps_taken, LLH_val


# ==================================
# MINI-FUNCTION TIL AT FINDE CLOSEST
# ==================================

def find_closest(range1,value):
    diff = np.abs(range1-value)
    min_indx = np.argmin(diff)
    return min_indx


# ==================================
# RASTERSCAN
# ==================================


def rasterscan(data, pdf, xrange, yrange, xtrue, ytrue, fig, ax, xlabel = None, ylabel = None, log_llh_fit = True, llh_fit = False, plot=True, plot_params=False, title = '2D Raster scan: Negative Ln-Likelihood values', delta=True, neg2log = True):

    '''One must define a figure before using this function.
    Returns a grid containing likelihood for each (xrange,yrange)-value-pair.
    '''

    LH_scanned = np.zeros((len(yrange),len(xrange)))
    
    for y in range(len(yrange)):
        for x in range(len(xrange)):

            # calculating likelihood for each (mu,std)-pair:
            if log_llh_fit:
                LH_scanned[y,x] = log_likelihood(data, pdf, [xrange[x],yrange[y]])  
            if llh_fit:
                LH_scanned[y,x] = likelihood(data, pdf, [xrange[x],yrange[y]])  
    
    
    max_indicies = np.unravel_index(np.argmin(LH_scanned), LH_scanned.shape)  # idx of best likelihood
    LH = LH_scanned[max_indicies]
    varx_bf = xrange[max_indicies[0]]  # best fit variable yrange
    vary_bf = yrange[max_indicies[1]]  # best fit variable xrange
    
    if plot:
        ax.set(xlabel = xlabel, ylabel = ylabel, title = title)
        
        if delta:
            LH_scanned -= np.min(LH_scanned)
            
        if delta and neg2log:
            LH_scanned -= np.min(LH_scanned)
            LH_scanned = 2*LH_scanned
            
        ax.imshow(LH_scanned)

        # ticks
        ax.set_xticks(np.arange(0,len(xrange))[::10])
        ax.set_xticklabels(np.round(xrange,2)[::10]);

        ax.set_yticks(np.arange(0,len(yrange))[::10])
        ax.set_yticklabels(np.round(yrange,2)[::10]);

        # colorbar
        im = ax.imshow(LH_scanned, vmin=np.min(LH_scanned), vmax = np.max(LH_scanned), cmap = 'ocean', alpha=0.6)
        cax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('LLH') 
        
        if delta:
            cbar.set_label('$\Delta$ LLH') 
            
        if delta and neg2log:
            cbar.set_label('-2$\Delta$ LLH')        
        
        
        # Likelihood fit 
        _ , par, _, _, _ = likelihood_fit(data, pdf, [xtrue,ytrue], ['x','y'])
            
        # Plotting best-fit parameter values
        xvar_bf_pos = find_closest(xrange,par[0])
        yvar_bf_pos = find_closest(yrange,par[1])
        ax.plot(xvar_bf_pos,yvar_bf_pos,marker='*',color='m',markersize=15, ls='none', label = 'Best-fit value')
        
        # Plotting contours (THESE ARE ONLY FOR -2DELTA LHH because of the parameters 1.15, 3.09 xxx multiplied by 2)
        x, y = np.arange(len(xrange)), np.arange(len(yrange))
        X,Y = np.meshgrid(x,y)
        colors = sns.color_palette("Paired", 10, desat=1)
        CS = ax.contour(X,Y, LH_scanned, [1.15,3.09,5.92]*2, colors = [colors[1], colors[2], colors[3]])  
        # ax.clabel(CS, inline=True, fontsize=10)  # to write values on the contour

        labels = [r'1$\sigma$', r'2$\sigma$',r'3$\sigma$']
        for i in range(len(labels)):
            CS.collections[i].set_label(labels[i])

            
        if plot_params:
            # Plotting true parameter values
            xvar_true_pos = find_closest(xrange, xtrue)
            yvar_true_pos = find_closest(yrange, ytrue)
            ax.plot(xvar_true_pos, yvar_true_pos,marker='*',color=colors[5],markersize=15, ls='none', label = 'True value')

        ax.legend(fontsize = 16, labelcolor='white');
    
    
    return LH_scanned, LH, vary_bf, varx_bf, max_indicies



# ==================================
# FROM TROELS PETERSEN FOR PLOTTING
# ==================================

def format_value(value, decimals):
    """ 
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """
    
    if isinstance(value, (float, np.float)):
        return f'{value:.{decimals}f}'
    elif isinstance(value, (int, np.integer)):
        return f'{value:d}'
    else:
        return f'{value}'


def values_to_string(values, decimals):
    """ 
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'. 
    """
    
    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f'{tmp[0]} +/- {tmp[1]}')
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """ Returns the length of the longest string in a list of strings """
    return len(max(s, key=len))


def nice_string_output(d, extra_spacing=5, decimals=3):
    """ 
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.  
    """
    
    names = d.keys()
    max_names = len_of_longest_string(names)
    
    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)
    
    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1 
        string += "{name:s} {value:>{spacing}} \n".format(name=name, value=value, spacing=spacing)
    return string[:-2]


def add_text_to_ax(x_coord, y_coord, string, ax, fontsize=12, color='k'):
    """ Shortcut to add text to an ax with proper font. Relative coords."""
    ax.text(x_coord, y_coord, string, family='monospace', fontsize=fontsize,
            transform=ax.transAxes, verticalalignment='top', color=color)
    return None



# ==================================
# PLOT NICE HISTROGRAM
# ==================================

def plot_hist(values, xmin, xmax, Nbins, ax, d_xy=[0.7, 0.5], xlabel=None, ylabel=None, title=None, color = 'g', label = 'Data', d_print=False, density = True, plot_err = False, ):
    
    """
    Function plotting a histogram of values with poisson errors
    
    BEWARE: One must define a figure BEFORE using this function, using ie. the line:
        fig, ax = plt.subplots(figsize = (10,6))
    """
    
    # Creating a classical histogram
    counts, bin_edges = np.histogram(values, bins=Nbins, range=(xmin, xmax), density = density)
    
    # Finding x, y and error on y (sy) given the histogram. Making sure bins are nonzero
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0]) 

    # Plotting
    ax.hist(values, bins=Nbins, alpha=0.3, range=(xmin, xmax), color = color, label = label, density = density, rwidth=0.92)
    if plot_err:
        ax.errorbar(x, y, sy, fmt='.k', ecolor='k', elinewidth=1, capsize=1, capthick=1, label = 'Counts with Possion errors')

    # Adding some values to plot
    if d_print == True:
        d = {'Entries':   len(values),
             'Mean':   np.mean(values),
             'Std':     np.std(values,ddof=1),
            }
        text = nice_string_output(d, extra_spacing=2, decimals=3)
        add_text_to_ax(*d_xy, text, ax, fontsize=12)
    
    ax.set(xlabel = xlabel, ylabel = ylabel, title=title)
    ax.grid(alpha=0.2)
    ax.legend()
    
    
    
    
# ==================================
# FROM SOPHIA
# ==================================
    
def correlation(x,y):
    'Correlation between two data sets'
    V_xy = 1/len(x) * sum((x - np.mean(x))*(y-np.mean(y)))
    rho_xy = V_xy / (np.std(x)*np.std(y))
    return rho_xy     

def correlation_plot(X, Y, N_bins, fig, ax, xlabel='X', ylabel='Y', d_xy=[0.05, 0.36]):
    'Plot 2d hist of two data sets'
    
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm

    counts, xedges, yedges, im = ax.hist2d(X, Y, bins=N_bins, cmap='coolwarm'); #, norm=LogNorm()
    divider = make_axes_locatable(ax)
    fig.colorbar(im, ax=ax)

    d = {'Entries': len(X),
         'Mean ' + xlabel : X.mean(),
         'Mean ' + ylabel : Y.mean(),
         'Std  ' + xlabel : X.std(ddof=1),
         'Std  ' + ylabel : Y.std(ddof=1),
         'Correlation' : np.cov(X,Y)[1,0]/(np.std(X)*np.std(Y))
        }

    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(*d_xy, text, ax, fontsize=14)
    
    
    
# ==================================
# FOR INTEGRATION
# ==================================
    
    
def integrate(y, x, xlower, xupper):
    '''Function taking the integral of y over x in the specific x-range given'''
    indices = np.searchsorted(x, [xlower, xupper])
    s = slice(indices[0], indices[1] + 1)             # creates a tuple and a slice object
    return np.trapz(y[s], x[s])



# ==================================
# ROC-CURVE FROM APPSTAT
# ==================================


# Calculate ROC curve from two histograms (hist1 is signal, hist2 is background):
def calc_ROC(hist1, hist2):

    # First we extract the entries (y values) and the edges of the histograms:
    # Note how the "_" is simply used for the rest of what e.g. "hist1" returns (not really of our interest)
    y_sig, x_sig_edges, _ = hist1 
    y_bkg, x_bkg_edges, _ = hist2
    
    # Check that the two histograms have the same x edges:
    if np.array_equal(x_sig_edges, x_bkg_edges):
        # kat: they have to be equal otherwise we cannot compare the bins of the differet data
        
        # Extract the center positions (x values) of the bins (both signal or background works - equal binning)
        x_centers = 0.5*(x_sig_edges[1:] + x_sig_edges[:-1])
        
        # Calculate the integral (sum) of the signal and background:
        integral_sig = y_sig.sum()
        integral_bkg = y_bkg.sum()
    
        # Initialize empty arrays for the True Positive Rate (TPR) and the False Positive Rate (FPR):
        TPR = np.zeros_like(y_sig) # True positive rate (sensitivity)
        FPR = np.zeros_like(y_sig) # False positive rate ()
        
        # Loop over all bins (x_centers) of the histograms and calculate TN, FP, FN, TP, FPR, and TPR for each bin:
        for i, x in enumerate(x_centers):  # i = [1,2,3,...], x = [x_center[0], x_center[1,...]]
            
            # The cut mask
            cut = (x_centers < x) # True/false array where it determines if the values in x_centers
                                  # are lower than x
            
            # True positive
            TP = np.sum(y_sig[~cut]) / integral_sig    # True positives
            FN = np.sum(y_sig[cut]) / integral_sig     # False negatives
            TPR[i] = TP / (TP + FN)                    # True positive rate
            
            #TP is the sum of all values higher than x. TN is the sum of all values lower than x.
            # TPR is the ratio, that gives the points on the ROC-curve. 
            
            # True negative
            TN = np.sum(y_bkg[cut]) / integral_bkg      # True negatives (background)
            FP = np.sum(y_bkg[~cut]) / integral_bkg     # False positives
            FPR[i] = FP / (FP + TN)                     # False positive rate            
            
        return FPR, TPR
    
    else:
        return('Signal and Background histograms have different bins and ranges')
    
    