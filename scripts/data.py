import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import datetime

import scanpy as sc
import scvelo as scv

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

def load_human(wt='/home/ngr4/project/scnd/data/processed/hum_ctrl.h5ad', 
               sca1='/home/ngr4/project/scnd/data/processed/hum_sca1.h5ad', 
               merge=False, velocity='/home/ngr4/project/scnd/data/human/', 
               custom_keep=True,
               verbose=True, **kwargs):
    '''
    Arguments:
      velocity (str or None): (optional, Default=None) enter loom filepath or leave as 
        None, which means no velocity data will be loaded
      custom_keep (bool): (optional, Default=True) slims down wt/sca1 data objects to
        retain just the gene and cell metadata, the imputed and normalized+transformed data
    '''
    if wt is not None:
        wt = sc.read(wt)
        if custom_keep:
            # retain only X, obs, var, imputed, and X_phate and X_umap slots
            sc.AnnData(X=wt.X, var=wt.var, obs=wt.obs, layers=wt.layers)
    if sca1 is not None:
        sca1 = sc.read(sca1)
        if custom_keep:
            # retain only X, obs, var, imputed, and X_phate and X_umap slots
            sca1 = sc.AnnData(X=sca1.X, obs=sca1.obs, var=sca1.var, layers=sca1.layers)
    if 'sca3' in kwargs:
        sca3 = sc.read(kwargs['sca3'])
    if merge:
        if not 'sca3' in kwargs:
            adata = wt.concatenate(sca1, batch_key='batch_adata', 
                                   batch_categories=['WT', 'SCA1'], 
                                   index_unique=None)
        else:
            adata = wt.concatenate([sca1, sca3], batch_key='batch_adata', 
                                   batch_categories=['WT', 'SCA1', 'SCA3'], 
                                   index_unique=None)
    if velocity is not None:
        loom_files = glob.glob(os.path.join(velocity,'*/*.loom'))
        if wt is None:
            loom_files = [i for i in loom_files if any([True if j in i else False for j in sca1.obs['batch'].unique()])]
        elif sca1 is None:
            loom_files = [i for i in loom_files if any([True if j in i else False for j in wt.obs['batch'].unique()])]
        sample_names = [os.path.split(os.path.split(loom_files[i])[0])[1] for i in range(len(loom_files))]
        if verbose:
            print('Loading looms...')
            for (name, f) in zip(sample_names, loom_files):
                print('  for {} @{}'.format(name, f))
        adata_looms = {}
        for i in range(len(loom_files)):
            start = time.time()
            if i == 0:
                adata_loom = scv.read_loom(loom_files[i], sparse=True, cleanup=True)
                adata_loom.var_names_make_unique()
            else:
                adata_looms[sample_names[i]] = scv.read_loom(loom_files[i], sparse=True, cleanup=True)
                adata_looms[sample_names[i]].var_names_make_unique()
        try:
            adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)
        except ValueError:
            adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)
        if verbose:
            print('looms loaded into sc.AnnData in {:.1f}-min'.format((time.time()-start)/60))
    if merge and velocity is not None:
        return scv.utils.merge(adata, adata_loom)
    elif merge and velocity is None:
        return adata
    if not merge:
        if wt is None and velocity is None:
            return sca1
        elif wt is None and velocity is not None:
            return scv.utils.merge(sca1, adata_loom)
        elif sca1 is None and velocity is None:
            return wt
        elif sca1 is None and velocity is not None:
            return scv.utils.merge(wt, adata_loom)
        elif not 'sca3' in kwargs and velocity is None:
            return wt, sca1
        elif 'sca3' in kwargs and velocity is None:
            return wt, sca1, sca3
    
    

def phate_from_adataknn(X, recalculate=False, bbknn=True, umap=False) :
    """Recalculate or use graph contained within adata for PHATE coords
    
    Args:
        X (AnnData): subsetted AnnData object
        recalculate (bool): (optional, Default: False) recalculate graph
        plot (ax object): optional. give ax object to plot in multiple for loop 
        save (str): optional. Save the plot with the full filepath indicated, otherwise return ax
    """
    if recalculate :
        # umap/louvain based off batch-balanced graph
        sc.tl.pca(X,n_comps=100)
        if bbknn:
            import bbknn
            bbknn.bbknn(X)
        else :
            sc.pp.neighbors(X, n_neighbors=30, n_pcs=100)

        # compute PHATE
        import phate
        import graphtools as gt
        from scipy import sparse
        G = gt.Graph(data=X.uns['neighbors']['connectivities']+sparse.diags([1]*X.shape[0], format='csr'),
                     precomputed='adjacency',
                     use_pygsp=True)
        G.knn_max = None

        phate_op = phate.PHATE(knn_dist='precomputed',
                               gamma=0,
                               n_jobs=-1)
        X.obsm['X_phate']=phate_op.fit_transform(G.K)
        if umap:
            sc.tl.umap(X)
    return X

def get_lt_server_fpaths(in_file='/home/ngr4/scratch60/llt_sequencing.txt',
                         out_file='/home/ngr4/project/scnd/data/processed/leon_tejwani_data_on_ruddle.csv'):
    os.system('ls -d /ycga-gpfs/sequencers/pacbio/gw92/10x/data_sent/llt26/*/*cellranger > {}'.format(in_file))
    df = pd.DataFrame()
    with open(in_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fpath, fname = os.path.split(line.strip())
            fname, _ = fname.split('_cellranger')
            df = df.append(pd.DataFrame({'fpath':fpath,
                                         'fname':fname}, index=[0]), 
                           ignore_index=True)
    if out_file is not None:
        df.to_csv(out_file)
    return None

def save_adata(adata, out_file, verbose=True):
    if verbose:
        print(adata)
    adata.write(out_file)
    if verbose:
        print('\nadata saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    return None

def merge_imputed(adata_out=None, **kwargs):
    if kwargs == {}:
        # assume files
        kwargs['ctrl'] = '/home/ngr4/project/scnd/data/processed/hum_wtimp_210928.h5ad'
        kwargs['sca1'] = '/home/ngr4/project/scnd/data/processed/hum_sca1imp_210928.h5ad'
    adatas = {}
    for i, (k, v) in enumerate(kwargs.items()):
        adatas[k] = sc.read(v)
    # concatenate, assume at least 2
    batch_names = list(adatas.keys())
    adata = adatas[batch_names[0]].concatenate(*[adatas[k] for k in batch_names[1:]],
                                       batch_key='source',
                                       batch_categories=batch_names)
    if adata_out is not None:
        save_adata(adata, adata_out)
    return adata

def load_human_redo(adata_file='/home/ngr4/project/scnd/data/processed/hum_210928.h5ad', imputed_adata='/home/ngr4/project/scnd/data/processed/hum_imputed_210928.h5ad', imputed=False):
    if imputed:
        # could do the merge here
        adata = sc.read(imputed_adata)
    else:
        adata = sc.read(adata_file)
    return adata

def load_annotated_hum_redo(adata_file='/home/ngr4/project/scnd/data/processed/hum_imp_211006.h5ad'):
    return sc.read(adata_file)

def merge_adata_loom(adata, loom_fp='/home/ngr4/project/scnd/data/human/rnavel/redo/', verbose=True):

        loom_files = glob.glob(os.path.join(loom_fp,'*/*.loom'))
        sample_names = [os.path.split(os.path.split(f)[0])[1] for f in loom_files]
        if verbose:
            print('Loading looms...')
            for (name, f) in zip(sample_names, loom_files):
                print('  for {} @{}'.format(name, f))
        adata_looms = {}
        for i, f in enumerate(loom_files):
            print('  ... loading {} @{}'.format(sample_names[i], f))
            start = time.time()
            if i == 0:
                adata_loom = scv.read_loom(f, sparse=True, cleanup=True)
                adata_loom.var_names_make_unique()
            else:
                adata_looms[sample_names[i]] = scv.read_loom(f, sparse=True, cleanup=True)
                adata_looms[sample_names[i]].var_names_make_unique()
        try:
            adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)
        except ValueError:
            adata_loom = adata_loom.concatenate(*adata_looms.values(), batch_categories=sample_names)
        if verbose:
            print('looms loaded into sc.AnnData in {:.1f}-min'.format((time.time()-start)/60))
        return scv.utils.merge(adata, adata_loom)
    
def load_mouse_imputed_revision(merge=False, **kwargs):
    '''Replace sc.AnnData.X slot with scAnnData.layers['imputed']
    '''
    # load imputed data
    wt = sc.read('/home/ngr4/project/scnd/data/processed/mouse_wt_imputed.h5ad')
    if 'add_md' in kwargs:
        wt.obs = wt.obs.merge(kwargs['add_md'], left_index=True, right_index=True)
    mut = sc.read('/home/ngr4/project/scnd/data/processed/mouse_sca1_imputed.h5ad')
    if 'add_md' in kwargs:
        mut.obs = mut.obs.merge(kwargs['add_md'], left_index=True, right_index=True)
    if merge:
        adata = wt
        del wt
        adata = wt.concatenate(mut, batch_key='imp_source', index_unique=None)
        return adata
    else:
        return wt, mut
