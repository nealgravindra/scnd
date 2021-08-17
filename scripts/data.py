import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import scanpy as sc

# settings
plt.rc('font', size = 9)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['text.usetex']=False
plt.rcParams['legend.frameon']=False
plt.rcParams['axes.grid']=False
plt.rcParams['legend.markerscale']=1#0.5
plt.rcParams['savefig.dpi']=600
sns.set_style("ticks")

# non-imputed data
def load_mouse(file='/home/ngr4/project/scnd/data/processed/mouse_210726.h5ad'):
    return sc.read(file)

# load imputed data
def imp_goi_to_adata(adata, goi, return_imputed_adatas=False, **kwargs):
    '''Add imputed expression to adata.obs metadata for plotting.
    
    Assumes indices match in adata, wt, and mut.
    
    Arguments:
      adata (sc.AnnData)
      goi (list)
      
    Returns:
      adata: copy of adata with genes of interest imputed expression in adata.obs column with Gene (imputed)
    '''
    if not 'wt' in kwargs:
        wt = sc.read('/home/ngr4/project/scnd/data/processed/mouse_wt_imputed.h5ad')
    else:
        wt = kwargs['wt']
    if not 'mut' in kwargs:
        # option to not supply both
        mut = sc.read('/home/ngr4/project/scnd/data/processed/mouse_sca1_imputed.h5ad')
    else:
        mut = kwargs['mut']
    
    for i, g in enumerate(goi):
        wt.obs['{} (imputed)'.format(g)] = np.asarray(wt[:, g].layers['imputed']).flatten()
        mut.obs['{} (imputed)'.format(g)] = np.asarray(mut[:, g].layers['imputed']).flatten()
        adata.obs = adata.obs.merge(wt.obs['{} (imputed)'.format(g)].append(mut.obs['{} (imputed)'.format(g)]), 
                                    how='left', left_index=True, right_index=True)
    
    if return_imputed_adatas:
        return adata, wt, mut
    else:
        return adata
    
def load_mouse_imputed(key_layer_slot='imputed', **kwargs):
    '''Replace sc.AnnData.X slot with scAnnData.layers['imputed']
    '''
    # load imputed data
    if not 'only_mut' in kwargs:
        if not 'wt' in kwargs:
            wt = sc.read('/home/ngr4/project/scnd/data/processed/mouse_wt_imputed.h5ad')
        else:
            wt = kwargs['wt']    
        if key_layer_slot is not None:
            wt = sc.AnnData(X=wt.layers[key_layer_slot], obs=wt.obs, var=wt.var)
    if not 'only_wt' in kwargs:
        if not 'mut' in kwargs:
            # option to not supply both
            mut = sc.read('/home/ngr4/project/scnd/data/processed/mouse_sca1_imputed.h5ad')
        else:
            mut = kwargs['mut']
        if key_layer_slot is not None:
            mut = sc.AnnData(X=mut.layers[key_layer_slot], obs=mut.obs, var=mut.var)
    if 'only_wt' in kwargs:
        adata = wt
        del wt
    elif 'only_mut' in kwargs:
        adata = mut
        del mut
    else:
        adata = wt.concatenate(mut, batch_key='imp_source', index_unique=None)
        del wt, mut
    if 'add_md' in kwargs:
        adata.obs = adata.obs.merge(kwargs['add_md'], left_index=True, right_index=True)
    return adata


def load_md(file='/home/ngr4/project/scnd/results/adata.obs_210726.csv', colkey='ctype_ubcupdate'):
    return pd.read_csv(file, index_col=0)[colkey]

def aesthetics():
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

    cmap_genotype={'WT':'#010101',
                   'SCA1':'#ffd478'}
    
    return {'cmap_mouse_ctype':cmap_ctype, 'cmap_gt':cmap_genotype}

def load_interactors(file='/home/ngr4/project/scnd/results/atxn_interactors.csv'):
    interactors = []
    with open(file) as f:
        for line in f:
            gene = line.rstrip()
            gene = gene[0] + gene[1:].lower()
            interactors.append(gene)
    f.close()
    if False:
        pfp = os.path.split(file)[0]
        missing_interactors = pd.read_csv(os.path.join(pfp,'missing_interactors.csv'), header=None).iloc[:,[0,1]].dropna()
        missing_interactors = {missing_interactors.iloc[i,0]:missing_interactors.iloc[i,1] for i in range(missing_interactors.shape[0])}
        interactors = pd.Series(interactors).replace(missing_interactors).to_list()
    return interactors

