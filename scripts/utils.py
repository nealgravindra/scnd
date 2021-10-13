import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

import scanpy as sc

import sys
sys.path.append('/home/ngr4/project')
import scnd.scripts.data as scnddata
import bbknn
import phate

def ctype_plots(tdata, markersoi, plotfp=None, short_name=None, groupby='ctype'):
    # plot 1
    sc.pl.StackedViolin(tdata, 
                markersoi,
                groupby=groupby, 
                use_raw=False, 
                layer='imputed').savefig(
        os.path.join(plotfp, 'violin_hum_redo_{}.pdf'.format(short_name)),
    bbox_inches='tight', 
    dpi=300)

    # plot 2
    sc.pl.DotPlot(tdata, 
                markersoi,
                groupby=groupby, 
                use_raw=True).savefig(
        os.path.join(plotfp, 'dot_hum_redo_{}.pdf'.format(short_name)), 
    bbox_inches='tight', 
    dpi=300)

    # plot 3
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.scatterplot(x=tdata.obsm['X_umap'][:, 0],
                    y=tdata.obsm['X_umap'][:, 1],
                    hue=tdata.obs[groupby],
                    linewidth=0,
                    alpha=0.8,
                    s=1,
                    rasterized=True,
                    ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    for l in tdata.obs[groupby].unique():
        x, y = np.mean(tdata[tdata.obs[groupby]==l, :].obsm['X_umap'], 0)
        ax.text(x, y, l)
    fig.savefig(os.path.join(plotfp, 'umap_hum_redo_{}.pdf'.format(short_name)), bbox_inches='tight', dpi=600)