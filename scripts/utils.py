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
    
    
def phateumap(AnnData, 
              recalc_cluster=False, 
              recalc_pca=False, n_pcs=50,
              recalc_knn=False, BBkNN=True, k=30,
              recalc_umap=True, recalc_phate=True,
              decay=40,
              batch_key='batch',
              adata_out=None, t='auto',
              gamma=0):
    import bbknn
    import phate
    import graphtools as gt
    from scipy import sparse
    '''Recalculate embeedings and dump in the AnnData.obsm slot.
    
    Arguments:
      k (int): if BBkNN true, this indicates the number of neighbors per batch
    '''
    if recalc_pca:
        sc.tl.pca(AnnData, n_comps=n_pcs)
    if recalc_knn:
        if BBkNN:
            bbknn.bbknn(AnnData, 
                        n_pcs=n_pcs, 
                        batch_key=batch_key,
                        neighbors_within_batch=k) # knn//len(adata.obs[batch_key].unique()
        else:
            sc.pp.neighbors(AnnData, n_pcs=n_pcs, n_neighbors=k)
    if recalc_cluster:
        sc.tl.leiden(AnnData, resolution=3.0)
    if recalc_umap:
        sc.tl.umap(AnnData)

    if recalc_phate:
        G = gt.Graph(data=AnnData.uns['neighbors']['connectivities']+sparse.diags([1]*AnnData.shape[0],format='csr'),
                     precomputed='adjacency',
                     use_pygsp=True)
        G.knn_max = None

        phate_op = phate.PHATE(knn_dist='precomputed',
                               gamma=gamma,
                               decay=decay,
                               n_jobs=-1,
                               t=t,
                               random_state=42)
        AnnData.obsm['X_phate']=phate_op.fit_transform(G.K)
    
    if adata_out is not None:
        scnddata.save_adata(AnnData, adata_out)
    
    return AnnData
