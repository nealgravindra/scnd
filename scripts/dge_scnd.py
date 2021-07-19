import pandas as pd
import os
import warnings
import numpy as np
import time
import scprep
import scanpy as sc
from scipy import sparse
import sys

from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests

def mwu(X, Y, gene_names, correction=None, debug=False, verbose=False) :
    '''
    Benjamini-Hochberg correction implemented. 
    
    gene_names (list)
    
    if X,Y single gene expression array, input x.reshape(-1,1), y.reshape(-1,1)

    NOTE: get zeros sometimes because difference (p-value is so small)
    '''
    p=pd.DataFrame()
    print('... starting Mann-Whitney U w/Benjamini/Hochberg correction...\n')
    start = time.time()
    for i,g in enumerate(gene_names) :
        if i==np.round(np.quantile(np.arange(len(gene_names)),0.25)) :
            print('... 25% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.5)) :
            print('... 50% completed in {:.2f}-s'.format(time.time()-start))
        elif i==np.round(np.quantile(np.arange(len(gene_names)),0.75)) :
            print('... 75% completed in {:.2f}-s'.format(time.time()-start))
        p.loc[i,'Gene']=g
        if (tiecorrect(rankdata(np.concatenate((np.asarray(X[:,i]),np.asarray(Y[:,i])))))==0) :
            if debug :
                print('P-value not calculable for {}'.format(g))
            p.loc[i,'pval']=np.nan
        else :
            _,p.loc[i,'pval']=mannwhitneyu(X[:,i],Y[:,i]) # continuity correction is True
    print('\n... mwu computed in {:.2f}-s\n'.format(time.time() - start))
    if not debug :
        # ignore NaNs, since can't do a comparison on these (change numbers for correction)
        p_corrected = p.loc[p['pval'].notna(),:]
        if p['pval'].isna().any():
            if verbose:
                print('Following genes had NA p-val:')
                for gene in p['Gene'][p['pval'].isna()]:
                    print('  %s' % gene)
    else : 
        p_corrected = p
    new_pvals = multipletests(p_corrected['pval'], method='fdr_bh')
    p_corrected['pval_corrected'] = new_pvals[1]
    return p_corrected

def log2aveFC(X,Y,gene_names,AnnData=None) :
    '''not sensitivity to directionality due to subtraction

    X and Y full arrays, subsetting performed here

    `gene_names` (list): reduced list of genes to calc

    `adata` (sc.AnnData): to calculate reduced list. NOTE: assumes X,Y drawn from adata.var_names
    '''
    if not AnnData is None :
        g_idx = [i for i,g in enumerate(AnnData.var_names) if g in gene_names]
        fc=pd.DataFrame({'Gene':AnnData.var_names[g_idx],
                         'log2FC':np.log2(X[:,g_idx].mean(axis=0)) - np.log2(Y[:,g_idx].mean(axis=0))}) # returns NaN if negative value 
    else :
        fc=pd.DataFrame({'Gene':gene_names,
                         'log2FC':np.log2(X.mean(axis=0)) - np.log2(Y.mean(axis=0))})
    return fc

def dge_scnd(data_A, data_B, colnames, out_file=None):
    '''Differential expression analysis adopted for scnd project for gex of A and B groups. 
    
    Arguments:
      data_A (np.ndarray): data for group A vs. B comparisons
      colnames (list): names for columns indexed in A & B, e.g., gene names
      fname (str): (optional, Default='Apos') Default value indicates that positive EMD values
        indicate up in A relative to B.
      self-explanatory.
    '''
    dge = pd.DataFrame()
    print('\nstarting {}\n'.format(fname))
    start_t=time.time()

    print('  n_obs in grp A:{}'.format(data_A.shape[0]))
    print('  n_obs in grp B:{}\n'.format(data_B.shape[0]))

        p = mwu(data_A, data_B, colnanmes, debug=True) # directionality doesn't matter
        emd = scprep.stats.differential_expression(data_A,data_B,
                                                   measure = 'emd',
                                                   direction='both', 
                                                   gene_names=colnames,
                                                   n_jobs=-1)
        emd['Gene']=emd.index
        emd=emd.drop(columns='rank')
        fc = log2aveFC(data_A, data_B, colnames)
        gene_mismatch = fc['Gene'].isin(p['Gene'])
        if gene_mismatch.any():
            fc = fc.loc[gene_mismatch, :]
            warnings.warn('Warning: {} genes dropped due to p-val NA.'.format((gene_mismatch==False).sum()))
        dt = pd.merge(p, fc, how='left', on="Gene")
        gene_mismatch = emd['Gene'].isin(p['Gene'])
        if gene_mismatch.any():
            emd = emd.loc[gene_mismatch,:]
        dt = pd.merge(dt,emd,how='left',on='Gene')
        dt['log10 P adj']=(-1)*np.log10(dt['pval_corrected'])

    print('\n... DGE computed in {:.2f}-s'.format(time.time()-start_t))

    if out_file is not None:
        # save volcano plot data
        dt.to_csv(out_file,index=False)

    return dt
