'''
'hum_dge_sca3.py'

Summary:
    1. imputation by genotype and save each
    2. then DGE comparing, for each cell type,
        - ctrl vs. sca1
        - ctrl vs. sca3
        - sca1 vs. sca3
        
Dataset:
    Sample IDs:
        adata.obs['Sample'] = adata.obs.batch.astype(str)
        adata.obs['Sample'][adata.obs.batch=='1-Jan']='SCA1-1'
        adata.obs['Sample'][adata.obs.batch=='14']='CTRL-1'
        adata.obs['Sample'][adata.obs.batch=='1516']='CTRL-3'
        adata.obs['Sample'][adata.obs.batch=='2-Jan']='SCA1-2'
        adata.obs['Sample'][adata.obs.batch=='3-Jan']='SCA1-3'
        adata.obs['Sample'][adata.obs.batch=='409']='CTRL-2'
        adata.obs.loc[adata.obs.batch=='32', 'Sample'] = 'SCA3-1'
        adata.obs.loc[adata.obs.batch=='33', 'Sample'] = 'SCA3-2'
        adata.obs.loc[adata.obs.batch=='34', 'Sample'] = 'SCA3-3'
        adata.obs.loc[adata.obs.batch=='35', 'Sample'] = 'SCA3-4'
        adata.obs.loc[adata.obs.batch=='122', 'Sample'] = 'SCA1-4'
        adata.obs.loc[adata.obs.batch=='132', 'Sample'] = 'SCA1-5'
        adata.obs.loc[adata.obs.batch=='15162', 'Sample'] = 'CTRL-4'
        adata.obs.loc[adata.obs.batch=='4092', 'Sample'] = 'CTRL-5'

        # annotate genotype
        adata.obs['Condition'] = [i.split('-')[0] for i in adata.obs['Sample']]
'''

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


###############################################
# load
###############################################

pdfp = '/home/ngr4/project/scnd/data/processed/'
pfp = '/home/ngr4/project/scnd/results'

if True:
    adata = utils.load_adata(os.path.join(pdfp, 'hum_wsca3.h5ad'))
    
    
###############################################
# imputation w/MAGIC
###############################################

# CTRL
print('Starting imputation for {}\n'.format('CTRL'))
tic = time.time()

wt = adata[adata.obs['Condition']=='CTRL', :]
wt.obs['value'] = 0
sc.pp.pca(wt)
sc.external.pp.bbknn(wt, batch_key='batch', n_pcs=50)

G = gt.Graph(data=wt.obsp['connectivities']+sparse.diags([1]*wt.shape[0],format='csr'),
             precomputed='adjacency',
             use_pygsp=True)
G.knn_max = None

magic_op=magic.MAGIC().fit(X=wt.X,graph=G) # running fit_transform produces wrong shape
wt.layers['imputed']=magic_op.transform(wt.X, genes='all_genes')
del G

if True:
    wt.write(os.path.join(pdfp, 'hum_ctrl.h5ad'))

print('\n  imputation in {:.2f}-min'.format((time.time() - tic)/60))


# SCA1
print('\n Starting imputation for {}\n'.format('SCA1'))
tic = time.time()

mut = adata[adata.obs['Condition']=='SCA1', :]
mut.obs['value'] = 0
sc.pp.pca(mut)
sc.external.pp.bbknn(mut, batch_key='batch', n_pcs=50)


G = gt.Graph(data=mut.obsp['connectivities']+sparse.diags([1]*mut.shape[0],format='csr'),
             precomputed='adjacency',
             use_pygsp=True)
G.knn_max = None

magic_op=magic.MAGIC().fit(X=mut.X,graph=G) # running fit_transform produces wrong shape
mut.layers['imputed']=magic_op.transform(mut.X,genes='all_genes')
del G

if True:
    mut.write(os.path.join(pdfp, 'hum_sca1.h5ad'))

print('\n  imputation in {:.2f}-min'.format((time.time() - tic)/60))

# SCA3
print('\n Starting imputation for {}\n'.format('SCA3'))
tic = time.time()

sca3 = adata[adata.obs['Condition']=='SCA3', :]
sca3.obs['value'] = 0
sc.pp.pca(sca3)
sc.external.pp.bbknn(sca3, batch_key='batch', n_pcs=50)


G = gt.Graph(data=sca3.obsp['connectivities']+sparse.diags([1]*sca3.shape[0],format='csr'),
             precomputed='adjacency',
             use_pygsp=True)
G.knn_max = None

magic_op=magic.MAGIC().fit(X=sca3.X,graph=G) # running fit_transform produces wrong shape
sca3.layers['imputed']=magic_op.transform(sca3.X,genes='all_genes')
del G

if True:
    sca3.write(os.path.join(pdfp, 'hum_sca3.h5ad'))

print('\n  imputation in {:.2f}-min'.format((time.time() - tic)/60))

# clean up space
del adata

###############################################
# DGE, ctrl V sca1
###############################################

fname = 'hum_ctrlVsca1'

dge = pd.DataFrame()
print('\nstarting {}\n'.format(fname))
start_t=time.time()

group='ctype'
for i in wt.obs[group].unique() :
    start = time.time()

    print('  group: {}\n'.format(i))

    X = wt[((wt.obs[group]==i)), :].layers['imputed']
    X_mut = mut[((mut.obs[group]==i)), :].layers['imputed']

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
    dt['Cell type']=[i]*dt.shape[0]
    dt['nlog10pvalcorrected']=(-1)*np.log10(dt['pval_corrected'])

    dge = dge.append(dt, ignore_index=True)
    print('... computed in {:.2f}-s'.format(time.time()-start))
print('\nFinished {} in {:.2f}-min'.format(fname,(time.time()-start_t)/60))

if True :
    # save volcano plot data
    dge.to_csv(os.path.join(pfp,'dge_'+fname+'.csv'),index=False)
    
# cleanup 
del dge, emd, fc, dt, X, X_mut, p

###############################################
# DGE, ctrl V sca3
###############################################

fname = 'hum_ctrlVsca3'

dge = pd.DataFrame()
print('\nstarting {}\n'.format(fname))
start_t=time.time()

group='ctype'
for i in wt.obs[group].unique() :
    start = time.time()

    print('  group: {}\n'.format(i))

    X = wt[((wt.obs[group]==i)), :].layers['imputed']
    X_mut = sca3[((sca3.obs[group]==i)), :].layers['imputed']

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
    dt['Cell type']=[i]*dt.shape[0]
    dt['nlog10pvalcorrected']=(-1)*np.log10(dt['pval_corrected'])

    dge = dge.append(dt, ignore_index=True)
    print('... computed in {:.2f}-s'.format(time.time()-start))
print('\nFinished {} in {:.2f}-min'.format(fname,(time.time()-start_t)/60))

if True :
    # save volcano plot data
    dge.to_csv(os.path.join(pfp,'dge_'+fname+'.csv'),index=False)
    
# cleanup 
del dge, emd, fc, dt, X, X_mut, p

###############################################
# DGE, sca1 V sca3
###############################################

fname = 'hum_sca1Vsca3'

dge = pd.DataFrame()
print('\nstarting {}\n'.format(fname))
start_t=time.time()

group='ctype'
for i in mut.obs[group].unique() :
    start = time.time()

    print('  group: {}\n'.format(i))

    X = mut[((mut.obs[group]==i)), :].layers['imputed']
    X_mut = sca3[((sca3.obs[group]==i)), :].layers['imputed']

    X = np.asarray(X)
    X_mut = np.asarray(X_mut)

    print('    Ncells in X:{}'.format(X.shape[0]))
    print('    Ncells in X_mut:{}\n'.format(X_mut.shape[0]))

    p = utils.mwu(X,X_mut,mut.var_names) # directionality doesn't matter
    emd = scprep.stats.differential_expression(X_mut,X,
                                               measure = 'emd',
                                               direction='both', 
                                               gene_names=mut.var_names,
                                               n_jobs=-1)
    emd['Gene']=emd.index
    emd=emd.drop(columns='rank')
    fc = utils.log2aveFC(X_mut,X,mut.var_names.to_list())
    gene_mismatch = fc['Gene'].isin(p['Gene'])
    if gene_mismatch.any():
        fc = fc.loc[gene_mismatch,:]
        warnings.warn('Warning: {} genes dropped due to p-val NA.'.format((gene_mismatch==False).sum()))
    dt = pd.merge(p,fc,how='left',on="Gene")
    gene_mismatch = emd['Gene'].isin(p['Gene'])
    if gene_mismatch.any():
        emd = emd.loc[gene_mismatch,:]
    dt = pd.merge(dt,emd,how='left',on='Gene')
    dt['Cell type']=[i]*dt.shape[0]
    dt['nlog10pvalcorrected']=(-1)*np.log10(dt['pval_corrected'])

    dge = dge.append(dt, ignore_index=True)
    print('... computed in {:.2f}-s'.format(time.time()-start))
print('\nFinished {} in {:.2f}-min'.format(fname,(time.time()-start_t)/60))

if True :
    # save volcano plot data
    dge.to_csv(os.path.join(pfp,'dge_'+fname+'.csv'),index=False)
    
# cleanup 
del dge, emd, fc, dt, X, X_mut, p

print('\nFINISHED.')









    
