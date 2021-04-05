import pandas as pd
import os
import warnings
import graphtools as gt
import numpy as np
import time
import scprep
import datetime
import scanpy as sc
from scipy import sparse
import sys
sys.path.append('/home/ngr4/project/scripts/')
import utils
import magic

def graph_pp(AnnData, bbknn=True, k=30, n_pcs=50):
    sc.tl.pca(AnnData, n_comps=50)
    if bbknn:
        sc.external.pp.bbknn(AnnData, n_pcs=n_pcs, neighbors_within_batch=int(k/len(AnnData.obs['batch'].unique()))) # use default params
    else:
        sc.pp.neighbors(AnnData, n_pcs=n_pcs, n_neighbors=k)
    return AnnData

# assume if __name__ == '__main__' is True

###############################################
# load
###############################################

pdfp = '/home/ngr4/project/scnd/data/processed'
pfp = '/home/ngr4/project/scnd/results/'
sc.settings.figdir = pfp

wt = utils.load_adata(os.path.join(pdfp,'mouse_wt_imputed.h5ad'))
mut = utils.load_adata(os.path.join(pdfp,'mouse_sca1_imputed.h5ad'))

# downsize
wt = sc.AnnData(X=wt.X, obs=wt.obs, var=wt.var)
mut = sc.AnnData(X=mut.X, obs=mut.obs, var=mut.var)

###############################################
# imputation
###############################################

grand_t = time.time()
paramset = [(1, 45)]

for i, kt in enumerate(paramset):
    t = kt[0]
    k = kt[1]

    # WT
    print('Starting imputations\tt:{}\tk:{}'.format(t, k))
    tic = time.time()

    wt = graph_pp(wt, k=k)
    mut = graph_pp(mut, k=k)

    G = gt.Graph(data=wt.obsp['connectivities']+sparse.diags([1]*wt.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None

    magic_op=magic.MAGIC(t=t).fit(X=wt.X,graph=G) # running fit_transform produces wrong shape
    wt.layers['imputed']=magic_op.transform(wt.X, genes='all_genes')
    del G

    # MUT
    G = gt.Graph(data=mut.obsp['connectivities']+sparse.diags([1]*mut.shape[0],format='csr'),
                 precomputed='adjacency',
                 use_pygsp=True)
    G.knn_max = None

    magic_op=magic.MAGIC(t=t).fit(X=mut.X,graph=G) # running fit_transform produces wrong shape
    mut.layers['imputed']=magic_op.transform(mut.X,genes='all_genes')
    del G

    print('  imputation in {:.2f}-min'.format((time.time() - tic)/60))


    ###############################################
    # DGE
    ###############################################

    fname = 'magic_t{}_k{}'.format(t,k)

    dge = pd.DataFrame()
    print('\nstarting {}\n'.format(fname))
    start_t=time.time()


    group1 = 'ctype'
    group2 = 'timepoint'
    for c in ['Purkinje cell']:
        for tpoint in ['5wk', '12wk', '18wk', '24wk', '30wk']:
            X = wt[((wt.obs[group1]==c) & (wt.obs[group2]==tpoint)), :].layers['imputed']
            X_mut = mut[((mut.obs[group1]==c) & (mut.obs[group2]==tpoint)), :].layers['imputed']

            X = np.asarray(X)
            X_mut = np.asarray(X_mut)

            print('    Ncells in X:{}'.format(X.shape[0]))
            print('    Ncells in X_mut:{}\n'.format(X_mut.shape[0]))

            p = utils.mwu(X,X_mut,wt.var_names) # directionality doesn't matter
            emd = scprep.stats.differential_expression(X_mut,X,
                                                       measure = 'emd',
                                                       direction='both',
                                                       gene_names=wt.var_names,
                                                       n_jobs=-1)
            emd['Gene']=emd.index
            emd=emd.drop(columns='rank')
            fc = utils.log2aveFC(X_mut,X,wt.var_names.to_list())
            gene_mismatch = fc['Gene'].isin(p['Gene'])
            if gene_mismatch.any():
                fc = fc.loc[gene_mismatch,:]
                warnings.warn('Warning: {} genes dropped due to p-val NA.'.format((gene_mismatch==False).sum()))
            dt = pd.merge(p,fc,how='left',on="Gene")
            gene_mismatch = emd['Gene'].isin(p['Gene'])
            if gene_mismatch.any():
                emd = emd.loc[gene_mismatch,:]
            dt = pd.merge(dt,emd,how='left',on='Gene')
            dt[group1]=[c]*dt.shape[0]
            dt[group2]=[tpoint]*dt.shape[0]
            dt['nlog10pvalcorrected']=(-1)*np.log10(dt['pval_corrected'])

            dge = dge.append(dt, ignore_index=True)

    print('  dge computed in {:.2f}-s'.format(time.time()-start_t))

    if True :
        # save volcano plot data
        dge.to_csv(os.path.join(pfp,'dge_'+fname+'_v2.csv'),index=False)

    print('  ... through {:.1f}-% in {:.1f}-min'.format(100*(i)/(len(paramset)), (time.time() - grand_t)/60))

    # cleanup
    del dge, emd, fc, dt, X, X_mut, p
