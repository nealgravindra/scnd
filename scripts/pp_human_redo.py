'''
data (re)-loader. Hyperparams this time are w/ & w/o mitochondrial transcripts 
and batch effect correction method

Neal G. Ravindra, 16 Mar 2020

'''

import os, sys, glob, re, math, pickle
import phate,magic,meld
import graphtools as gt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time,random,datetime
from scipy import sparse
import scanpy as sc
import bbknn

def chk_files(data_folders):
    files_not_found = []
    for i in data_folders :
        if not os.path.exists(i) :
            files_not_found.append(i)
    if not files_not_found == [] :
        print('Folders not found...')
    else: 
        return True
    for j in files_not_found :
        print(j)
        raise IOError('Change path to data.')

def save_adata(adata, out_file, verbose=True):
    if verbose:
        print(adata)
    adata.write(out_file)
    if verbose:
        print('\nadata saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    return None

def cellranger2adata(data_folders, sample_str_pattern='(.*)_HNT', adata_out=None):

    start = time.time()
    adatas = {}
    running_cellcount = 0
    for i,folder in enumerate(data_folders) :
        animal_id = os.path.split(re.findall(sample_str_pattern, folder)[0])[1]
        print('... storing %s into dict (%d/%d)' % (animal_id, i+1, len(data_folders)))
        adatas[animal_id] = sc.read_10x_mtx(folder)
        running_cellcount += adatas[animal_id].shape[0]
        print('...     read {} cells; total: {} in {:.2f}-s'.format(adatas[animal_id].shape[0],running_cellcount,time.time()-start))
    batch_names = list(adatas.keys())
    print('\n... concatenating of {}-samples'.format(len(data_folders)))
    adata = adatas[batch_names[0]].concatenate(*[adatas[k] for k in batch_names[1:]],
                                               batch_categories = batch_names)
    print('Raw load in {:.2f}-min'.format((time.time() - start)/60))

    if adata_out is not None :
        save_adata(adata, adata_out)
    
    return adata
    
def pp(adata, doublets=True, adata_out=None):

    # filter cells/genes, transform
    sc.pp.calculate_qc_metrics(adata,inplace=True)
    mito_genes = adata.var_names.str.startswith('MT-')
    adata.obs['pmito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
    print('Ncells=%d have >10percent mt expression' % np.sum(adata.obs['pmito']>0.1))
    print('Ncells=%d have <200 genes expressed' % np.sum(adata.obs['n_genes_by_counts']<200))
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3) # filtering cells gets rid of some genes of interest
    adata = adata[adata.obs.pmito <= 0.1, :]
    adata.raw = adata
    if doublets:
        sc.external.pp.scrublet(adata)
    sc.pp.normalize_total(adata)
    sc.pp.sqrt(adata)
    if adata_out is not None:
        save_adata(adata, adata_out)
    return adata

def batch_correction(adata, adata_out=None, batch_key='Sample', knn=100, 
                     plot1_out=None, plot2_out=None):
    '''
    Arguments:
      plot*_out (bool/str/None): if False, don't make plots. Replace None with str to save.
    '''

    if plot1_out or plot1_out is None or type(plot1_out)==str:
        # batch effect plot
        tdata = adata
        sc.tl.pca(tdata,n_comps=100)
        sc.pp.neighbors(tdata,n_neighbors=30,n_pcs=100)
        sc.tl.umap(tdata)
        G = gt.Graph(data=tdata.uns['neighbors']['connectivities']+sparse.diags([1]*tdata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
        G.knn_max = None
                                  
        phate_op = phate.PHATE(knn_dist='precomputed',
                           gamma=0,
                           n_jobs=-1,
                           random_state=42)
        tdata.obsm['X_phate']=phate_op.fit_transform(G.K)

        fig, ax=plt.subplots(1, 3, figsize=(16, 4))

        sns.scatterplot(x=tdata.obsm['X_pca'][:, 0],
                        y=tdata.obsm['X_pca'][:, 1],
                              hue=adata.obs[batch_key],
                              legend=False,
                              ax=ax[0],
                              s = 1,
                              alpha=0.6,
                              rasterized=True,)
        sns.scatterplot(x=tdata.obsm['X_umap'][:, 0],
                        y=tdata.obsm['X_umap'][:, 1],
                              hue=adata.obs[batch_key],
                              legend=False,
                              ax=ax[1],
                              s = 1,
                              alpha=0.6,
                              rasterized=True,)
        sns.scatterplot(x=tdata.obsm['X_phate'][:, 0],
                        y=tdata.obsm['X_phate'][:, 1],
                              hue=adata.obs[batch_key],
                              legend=True,
                              ax=ax[2],
                              s = 1,
                              alpha=0.6,
                              rasterized=True,)
        ax[0].set_xlabel('PCA1')
        ax[0].set_ylabel('PCA2')
        ax[1].set_xlabel('UMAP1')
        ax[1].set_ylabel('UMAP2')
        ax[2].set_xlabel('PHATE1')
        ax[2].set_ylabel('PHATE2')

        fig.tight_layout()

        if type(plot1_out)==str:
            fig.savefig(plot1_out, dpi=600, bbox_inches='tight')
        del tdata
    
    start = time.time()
    print('Starting embeddings and batch effect correcftion...')
    sc.tl.pca(adata, n_comps=100)
    bbknn.bbknn(adata, 
                         n_pcs=100, 
                         batch_key=batch_key,
                         neighbors_within_batch=knn//len(adata.obs[batch_key].unique()),
                        ) 
    sc.tl.leiden(adata, resolution=3.0)
    sc.tl.umap(adata)

    if adata_out is not None:
        save_adata(adata, adata_out)

    # phate
    G = gt.Graph(data=adata.uns['neighbors']['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None
                                  
    phate_op = phate.PHATE(knn_dist='precomputed',
                           gamma=0,
                           n_jobs=-1,
                           random_state=42)
    adata.obsm['X_phate']=phate_op.fit_transform(G.K)

    if adata_out is not None:
        save_adata(adata, adata_out)
    
    print('\n  BB-kNN batch crct + UMAP + PHATE in {:.1f}-min'.format((time.time() - start)/60))
    
    if plot2_out or plot2_out is None or type(plot2_out)==str:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))

        sns.scatterplot(x=adata.obsm['X_pca'][:, 0],
                        y=adata.obsm['X_pca'][:, 1],
                        hue=adata.obs[batch_key],
                        linewidth=0, s=1, alpha=0.6,
                        rasterized=True, legend=False, 
                        ax=ax[0],
                        )
        ax[0].set_xlabel('PCA1')
        ax[0].set_ylabel('PCA2')
        sns.scatterplot(x=adata.obsm['X_umap'][:, 0],
                        y=adata.obsm['X_umap'][:, 1],
                        hue=adata.obs[batch_key],
                        linewidth=0, s=1, alpha=0.6,
                        rasterized=True, legend=False, 
                        ax=ax[1],
                        )
        ax[1].set_xlabel('UMAP1')
        ax[1].set_ylabel('UMAP2')
        sns.scatterplot(x=adata.obsm['X_phate'][:, 0],
                        y=adata.obsm['X_phate'][:, 1],
                        hue=adata.obs[batch_key],
                        linewidth=0, s=1, alpha=0.6,
                        rasterized=True, legend=True, 
                        ax=ax[2],
                        )
        ax[2].set_xlabel('PHATE1')
        ax[2].set_ylabel('PHATE2')
        fig.tight_layout()

        if type(plot2_out)==str:
            fig.savefig(plot2_out, bbox_inches='tight', dpi=600)

    return adata

def magic_impute(adata, adata_out=None, batch_key='Sample', t=1):
    tic = time.time()
    def graph_pp(AnnData, use_bbknn=True, k=30, n_pcs=100, batch_key=batch_key):
        sc.tl.pca(AnnData, n_comps=n_pcs)
        if use_bbknn:
            bbknn.bbknn(AnnData,
                        n_pcs=n_pcs,
                        neighbors_within_batch=k // len(adata.obs[batch_key].unique()))
        else:
            sc.pp.neighbors(AnnData, n_pcs=n_pcs, n_neighbors=k)
        return AnnData
    adata = graph_pp(adata)
    G = gt.Graph(data=adata.obsp['connectivities']+sparse.diags([1]*adata.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True,)
    G.knn_max = None

    magic_op=magic.MAGIC(t=t).fit(X=adata.X, graph=G)
    adata.layers['imputed']=magic_op.transform(adata.X, genes='all_genes')
    print('\n... imputation in {:.1f}-min for {}-cells x {}-genes'.format((time.time() - tic)/60, *adata.shape))
    if adata_out is not None:
        save_adata(adata, adata_out)
    return adata

if __name__ == '__main__':
    total = time.time()

    # settings
    plt.rc('font', size = 8)
    plt.rc('font', family='sans serif')
    plt.rcParams['legend.frameon']=False
    plt.rcParams['axes.grid']=False
    plt.rcParams['legend.markerscale']=0.5
    sc.set_figure_params(dpi=300, dpi_save=600,
                         frameon=False,
                         fontsize=8)
    plt.rcParams['savefig.dpi'] = 600

    ####
    # files
    ####
    pfp = '/home/ngr4/project/scnd/results/'
    pdfp = '/home/ngr4/scratch60/scnd/data/processed/human_cellranger'
    data_folders = [
        '122_HNT', '14_HNT', 
        '1516_HNT', '2-Jan_HNT', 
        '4092_HNT', '132_HNT', 
        '15162_HNT', '1-Jan_HNT', 
        '3-Jan_HNT', '409_HNT',
        ]
    data_folders = [os.path.join(i, 'outs/filtered_feature_bc_matrix/') for i in data_folders]
    adata_out = '/home/ngr4/project/scnd/data/processed/hum_210920.h5ad'
    ####

    sc.settings.figdir = pfp
    data_folders = [os.path.join(pdfp, i) for i in data_folders]

    if chk_files(data_folders):
        print('Loading {} samples'.format(len(data_folders)))
    adata = cellranger2adata(data_folders, adata_out=adata_out)
    
    ####
    # rename batches MANUALLY
    ####
    adata.obs['Sample'] = adata.obs.batch.astype(str)
    adata.obs.loc[adata.obs.batch=='1-Jan', 'Sample'] = 'SCA1-1'
    adata.obs.loc[adata.obs.batch=='14', 'Sample'] = 'CTRL-1'
    adata.obs.loc[adata.obs.batch=='1516', 'Sample'] = 'CTRL-4'
    adata.obs.loc[adata.obs.batch=='2-Jan', 'Sample'] = 'SCA1-2'
    adata.obs.loc[adata.obs.batch=='3-Jan', 'Sample'] = 'SCA1-3'
    adata.obs.loc[adata.obs.batch=='409', 'Sample'] = 'CTRL-2'
    adata.obs.loc[adata.obs.batch=='15162', 'Sample'] = 'CTRL-5'
    adata.obs.loc[adata.obs.batch=='4092', 'Sample'] = 'CTRL-3'
    adata.obs.loc[adata.obs.batch=='122', 'Sample'] = 'SCA1-4'
    adata.obs.loc[adata.obs.batch=='132', 'Sample'] = 'SCA1-5'  
    ####

    adata.obs['genotype'] = 'None'
    adata.obs.loc[[True if 'SCA1' in i else False for i in adata.obs['Sample']], 'genotype'] = 'SCA1'
    adata.obs.loc[[True if 'CTRL' in i else False for i in adata.obs['Sample']], 'genotype'] = 'CTRL'
    adata = pp(adata, adata_out=adata_out)
    
    ####
    # plot names
    ####
    plot1 = os.path.join(pfp, 'hum_redo_pre-batchcrct.pdf')
    plot2 = os.path.join(pfp, 'hum_redo_post-batchcrct.pdf')
    ####

    adata = batch_correction(adata, adata_out=adata_out,
                             plot1_out=plot1,
                             plot2_out=plot2)

    ####
    # impute adata names
    ###
    wt_out = '/home/ngr4/project/scnd/data/processed/hum_wtimp_210920.h5ad'
    mut_out = '/home/ngr4/project/scnd/data/processed/hum_sca1imp_210920.h5ad'
    ####

    wt = magic_impute(adata[adata.obs['genotype']=='CTRL', :], adata_out=wt_out)
    mut = magic_impute(adata[adata.obs['genotype']=='SCA1', :], adata_out=mut_out)

    print('Pre-processing dataset took {:.2f}-min'.format((time.time() - total)/60))
