import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
import os
import posixpath
from functools import partial
from sklearn.model_selection import RepeatedKFold
import sklearn.linear_model as skl
import scipy as sp
import matplotlib as mpl
import random

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from IPython.display import display

# set all single line variables to be displayed, not just the last line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def set_plot_formatting():
    # set up plot formatting
    SMALL_SIZE = 12#12  15
    MEDIUM_SIZE = 15#15  18
    BIGGER_SIZE = 25#18  25

    # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    # plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

def grayscale_version(color):
    # converts colors to how they will look in gray scale
    conversion = np.array([0.299, 0.587, 0.114])

    grayscale = np.repeat(np.matmul(conversion, color), 3)

    return grayscale

def check_colors(colors, linewidth):
    # plots the colors to be used

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    for i in range(len(colors)):
        _ = ax.plot([0, .5], [i, i], color=colors[i], linewidth=linewidth)
        _ = ax.plot([.5, 1], [i, i], color=grayscale_version(colors[i][:-1]), linewidth=linewidth)
        _ = ax.axis('off')