import pandas as pd
import os
import glob
import pickle
import phate
import meld
import time
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import scanpy as sc
import math
from scipy import sparse

import sys
sys.path.append('/home/ngr4/project/')
from scnd.scripts import utils as scndutils
from scnd.scripts import data as scnddata

from scipy.stats import zscore, binned_statistic
from scipy.ndimage import gaussian_filter1d

# settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['text.usetex']=False
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=0.5
sc.set_figure_params(dpi=300,dpi_save=600,
                     frameon=False,
                     fontsize=9)
plt.rcParams['savefig.dpi']=600
sc.settings.verbosity=2
sc._settings.ScanpyConfig.n_jobs=-1
sns.set_style("ticks")

def meld_pseudotime(AnnData, goi, one_df=False):
    if AnnData.shape[0] > 5000:
        AnnData = AnnData[AnnData.obs.sample(5000, replace=True).index, :]
        AnnData.obs_names_make_unique() 
    AnnData = AnnData.copy() # to allow subsetting?
    sc.pp.pca(AnnData)
    sc.pp.neighbors(AnnData, n_pcs=50)

    # MELD
    G = gt.Graph(data=AnnData.obsp['connectivities']+sparse.diags([1]*AnnData.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None
    AnnData.obs['res_t']=-1
    AnnData.obs.loc[AnnData.obs['timepoint']=='12wk', 'res_t'] = -0.5
    AnnData.obs.loc[AnnData.obs['timepoint']=='18wk', 'res_t'] = 0
    AnnData.obs.loc[AnnData.obs['timepoint']=='24wk', 'res_t'] = 0.5
    AnnData.obs.loc[AnnData.obs['timepoint']=='30wk', 'res_t'] = 1
    AnnData.obs['ees_t'] = meld.MELD().fit_transform(G=G, RES=AnnData.obs['res_t'])
    AnnData.obs['ees_t'] = (AnnData.obs['ees_t'] - AnnData.obs['ees_t'].min()) / (AnnData.obs['ees_t'].max() - AnnData.obs['ees_t'].min())

    # strata
    X = pd.DataFrame(AnnData[:, goi].layers['imputed'], columns=goi, index=AnnData.obs.index.to_list())
    y = AnnData.obs['ees_t'].to_numpy()
    
    if one_df:
        # collate
        X['genotype'] = AnnData.obs['genotype'].to_list()
        X['timepoint'] = AnnData.obs['timepoint'].to_list()
        X['Pseudotime'] = y
        X['ctype'] = AnnData.obs['ctype_ubcupdate'].to_list()
        return X
    else:
        return X, y

def dyn_gene_diff(WT, MUT, goi='all', gkey='ctype_ubcupdate', gcname=None,
                  pfp='/home/ngr4/project/scnd/results/dyngoi/',
                  normtype='ctype',
                  ntimebins=20, verbose=True, out_file=None,
                  mitoprefix='mt-', exclude_genes=['Xist', 'Tsix', 'Eif2s3y']):
    '''
    Arguments:
      gkey (str) [optional, Default='ctype_ubcupdate']: .obs[key] specification
        for grouping.
      gcname (str or None) [optional, Default=None]: if None, loop through
        all unique in c grouping. 
      normtype (str or None) [optional, Default=gtype_ctype]: standardize gex across genotype AND ctype groups. 
        Other options are None and ctype.
    ''' 
    os.makedirs(pfp, exist_ok=True)
    if goi == 'all':
        goi = WT.var_names.to_list()
        mito_genes = [i for i in WT.var_names if i.startswith(mitoprefix)]
        exclude_genes = exclude_genes + mito_genes # this is kind of ignored with goi calls
        goi = [i for i in goi if i not in exclude_genes]
    results = pd.DataFrame()
    
    def ave_normedgex_pertbin_peradata(wt_gc, mut_gc, goi=goi, normtype=normtype, ntimebins=ntimebins):
        # general, so automate
        X_wt, y_wt = meld_pseudotime(wt_gc, goi)
        X_mut, y_mut = meld_pseudotime(mut_gc, goi)
        y = np.concatenate((y_wt, y_mut))

        # binning
        _, _, time_groupings = binned_statistic(y, y, bins=ntimebins)
        
        if normtype is not None:
            print(normtype, 'here')
            if normtype=='gtype_ctype':
                X_wt = pd.DataFrame(zscore(X_wt), index=X_wt.index, columns=X_wt.columns)
                X_mut = pd.DataFrame(zscore(X_mut), index=X_mut.index, columns=X_mut.columns)
            elif normtype=='ctype':
                X = X_wt.append(X_mut)
                X = pd.DataFrame(zscore(X), index=X.index, columns=X.columns)
                X_wt = X.loc[X_wt.index, :]
                X_mut = X.loc[X_mut.index, :]
            else:
                raise NotImplementedError
        X_wt['time_grouping'] = time_groupings[0:y_wt.shape[0]]
        X_wt = X_wt.groupby('time_grouping').mean().T
        X_mut['time_grouping'] = time_groupings[y_wt.shape[0]:]
        X_mut = X_mut.groupby('time_grouping').mean().T
        return X_wt, X_mut
    
    def make_plots(X1, X2, gcname, dX_df, out_file=out_file, ntimebins=ntimebins):
        '''
        Arguments:
          X1: should be a dataframe with time_grouping in col index and values of elements
            are average GEX values for gene indicated in index
          X2: same as X1 but for mut AnnData obj
        '''
        cm = plt.get_cmap('magma')
        timecolors = [cm(1.*i/ntimebins) for i in range(ntimebins)]
        
        # hmap1, WT
        p = sns.clustermap(
            X1,
            pivot_kws=None,
            method='average',
            metric='euclidean',
            z_score=None,
            standard_scale=None,
            row_cluster=True,
            col_cluster=False,
            row_linkage=None,
            figsize=(4, 20),
            col_linkage=None,
            row_colors=None,
            col_colors=timecolors,
            mask=None,
            dendrogram_ratio=0.2,
            colors_ratio=0.03,
            cbar_pos=(1, 0, 0.1, 0.15),
            tree_kws=None,
            cmap='RdYlBu_r',
            yticklabels=True,
            xticklabels=False,
            vmax=1, vmin=-1,
        )
        p.ax_heatmap.set_xlabel('')
        p.savefig(os.path.join(pfp, 'hmap_gexVtbin_wt_{}.pdf'.format(gcname)), bbox_inches='tight')
        
        # hmap2, MUT
        p = sns.clustermap(
            X2,
            pivot_kws=None,
            method='average',
            metric='euclidean',
            z_score=None,
            standard_scale=None,
            row_cluster=True,
            col_cluster=False,
            row_linkage=None,
            figsize=(4, 20),
            col_linkage=None,
            row_colors=None,
            col_colors=timecolors,
            mask=None,
            dendrogram_ratio=0.2,
            colors_ratio=0.03,
            cbar_pos=(1, 0, 0.1, 0.15),
            tree_kws=None,
            cmap='RdYlBu_r',
            yticklabels=True,
            xticklabels=False,
            vmax=1, vmin=-1,
        )
        p.ax_heatmap.set_xlabel('')
        p.savefig(os.path.join(pfp, 'hmap_gexVtbin_mut_{}.pdf'.format(gcname)), bbox_inches='tight')
        
        # hmap3, DIFF
        dX  = X2.subtract(X1) # MUT - WT
        p_diff = sns.clustermap(
                dX,
                pivot_kws=None,
                method='average',
                metric='euclidean',
                z_score=None,
                standard_scale=None,
                row_cluster=True,
                col_cluster=False,
                row_linkage=None,
                figsize=(4, 20), 
                col_linkage=None,
                row_colors=None,
                col_colors=timecolors,
                mask=None,
                dendrogram_ratio=0.2,
                colors_ratio=0.03,
                cbar_pos=(1, 0, 0.1, 0.15),
                tree_kws=None,
                cmap='RdYlBu_r',
                yticklabels=True,
                xticklabels=False,
                vmax=1, vmin=-1,
            )
        p_diff.ax_heatmap.set_xlabel('')
        p_diff.savefig(os.path.join(pfp, 'hmap_gexVtbin_DIFF_{}.pdf'.format(gcname)), bbox_inches='tight')
        
        # regplot
        x_diff_melted = dX.T.reset_index().melt(id_vars='time_grouping', var_name='GOI', value_name='mut-wt')
        fig, ax = plt.subplots(1,1, figsize=(4,3))
        p = sns.regplot('time_grouping', 'mut-wt', data=x_diff_melted, 
                    x_estimator=None, x_bins=None, x_ci='ci', scatter=True, 
                    fit_reg=True, ci=95, n_boot=1000, units=None, seed=None, 
                    order=1, logistic=False, lowess=True, robust=False, logx=False, x_partial=None, 
                    y_partial=None, truncate=True, dropna=True, x_jitter=0.5, y_jitter=None, 
                    label=None, color='gray', marker='o', scatter_kws={'s':1, 'alpha':0.6, 'lw':0}, line_kws=None, ax=ax)
        fig.savefig(os.path.join(pfp, 'regplot_mut-wtVtbin_{}.pdf'.format(gcname)), bbox_inches='tight')
    
        # histogram + smoothed line
        fig, ax = plt.subplots(1,1, figsize=(4,3))
        z = dX.abs().mean(axis=0).reset_index()
        z = z.rename(columns={0:'ave_mut-wt'})
        sns.barplot(x='time_grouping', y='ave_mut-wt', data=z, color='gray', ax=ax)
        yprime = gaussian_filter1d(
            z['ave_mut-wt'],
            2,
            axis=-1,
            order=0,
            output=None,
            mode='nearest',
            cval=0.0,
            truncate=4.0,
        )
        ax.plot(z['time_grouping']-1, yprime, color='gray')
        fig.savefig(os.path.join(pfp, 'hist_mut-wtABSDIFF_{}f'.format(gcname)), bbox_inches='tight')

        dX = dX.reset_index()
        dX['group'] = gcname
        dX_df = dX_df.append(dX, ignore_index=True)
        # intermediate save
        if out_file is not None:
            dX_df.to_csv(out_file)
        return dX_df
            
    if gcname is None:
        for ii, gc in enumerate(np.sort(WT.obs[gkey].unique())):
            if gc=='Granule cell': 
                # sample with replacement
                nsample_idx = WT.obs.loc[(WT.obs[gkey]==gc), :].sample(5000).index.to_list()
                wt = WT[(WT.obs[gkey]==gc) & (WT.obs.index.isin(nsample_idx))].copy()
                nsample_idx = MUT.obs.loc[(MUT.obs[gkey]==gc), :].sample(5000).index.to_list()
                mut = MUT[(MUT.obs[gkey]==gc) & (MUT.obs.index.isin(nsample_idx))].copy()
            else:
                wt = WT[(WT.obs[gkey]==gc), :].copy()
                mut = MUT[(MUT.obs[gkey]==gc), :].copy()
                
            X_wt, X_mut = ave_normedgex_pertbin_peradata(wt, mut)
            results = make_plots(X_wt, X_mut, gc, results)
            
    else:
        wt = WT[(WT.obs[gkey]==gcname), :].copy()
        mut = MUT[(MUT.obs[gkey]==gcname), :].copy()
        
        X_wt, X_mut = ave_normedgex_pertbin_peradata(wt, mut)
        results = make_plots(X_wt, X_mut, gcname, results)
    
    return results 

def diffplot_allgroups(x, ntimebins=20, sigma=2, gkey='group', pfp='/home/ngr4/project/scnd/results/'):
    cm = plt.get_cmap('magma')
    timecolors = [cm(1.*i/ntimebins) for i in range(ntimebins)]
    diffs = pd.DataFrame()
    for i in x[gkey].unique():
        x_diff = x.loc[x[gkey]==i, [ii for ii in range(1, ntimebins+1)]+['index']].set_index('index')
        

        z = x_diff.abs().mean(axis=0).reset_index().rename(columns={'time_grouping':'time'})
        z['time'] = z['time'].astype(int)
        z = z.rename(columns={0:'ave_mut-wt'})

        # smoothed
        yprime = gaussian_filter1d(
            z['ave_mut-wt'],
            sigma,
            axis=-1,
            order=0,
            output=None,
            mode='nearest',
            cval=0.0,
            truncate=4.0,
        )

        dt = pd.DataFrame({'time':z['time'] - 1, 'ave_mut-wt':yprime})
        dt[gkey] = i

        diffs = diffs.append(dt, ignore_index=True)
        
    # aesthetics
    cmap_ctype={'Granule cell': '#FAC18A',
                'Unipolar brush cell': '#BA61BA',
                'Purkinje cell': '#EE5264',
                'GABAergic interneuron 1': '#F9EBAE',
                'GABAergic interneuron 2': '#88BB92',
                'GABAergic interneuron 3': '#46A928',
                'Astrocyte': '#F9AEAE',
                'Bergmann glia': '#AEB7F9',
                'Oligodendrocyte progenitor cell': '#F1815F',
                'Oligodendrocyte': '#75A3B7',
                'Microglia': '#AC5861',
                'Pericyte': '#2D284B',
                'Endothelial cell': '#1C67EE',
                'Deep cerebellar nuclei': '#aaaaaa'}

    fig, ax = plt.subplots(1,1, figsize=(5,4))
    sns.lineplot(x='time', y='ave_mut-wt', hue=gkey, size=None, 
                     style=None, data=diffs, palette=cmap_ctype, hue_order=None, 
                     hue_norm=None, sizes=None, size_order=None, size_norm=None, 
                     dashes=True, markers=None, style_order=None, 
                     units=None, estimator='mean', ci=95, n_boot=1000, seed=None, sort=True, 
                     err_style='band', err_kws=None, legend='brief', ax=ax, lw=3)
    ax.legend(bbox_to_anchor=(1.1,1)).set_title('')
    ax.set_xticks([0,19])
    fig.savefig(os.path.join(pfp, 'dyn_diffs_ctypes.pdf'), bbox_inches='tight')
    return diffs

if __name__ == '__main__':

    wtt, mutt = scnddata.load_mouse_imputed_revision(add_md=scnddata.load_md())
    dX_df = dyn_gene_diff(wtt, mutt)



        

            

