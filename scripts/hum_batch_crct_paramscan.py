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

import sys
sys.path.append('/home/ngr4/project/')
from scnd.scripts import pp_human_redo as scndpp

def init_load(pdfp, 
              adata_out='/home/ngr4/project/scnd/data/processed/hum_210927.h5ad'):
    data_folders = [
        '122_HNT', '14_HNT', 
        '1516_HNT', '2-Jan_HNT', 
        '4092_HNT', '132_HNT', 
        '15162_HNT', '1-Jan_HNT', 
        '3-Jan_HNT', '409_HNT',
        ]
    data_folders = [os.path.join(i, 'outs/filtered_feature_bc_matrix/') for i in data_folders]
    data_folders = [os.path.join(pdfp, i) for i in data_folders]

    if scndpp.chk_files(data_folders):
        print('Loading {} samples'.format(len(data_folders)))
    adata = scndpp.cellranger2adata(data_folders, adata_out=adata_out)
    
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
    return adata

def load_pp_hum_adata(
    adata_file='/home/ngr4/project/scnd/data/processed/hum_210927.h5ad'):
    return sc.read(adata_file)

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
    pfp = '/home/ngr4/project/scnd/results/hum_redo_batchcrct'
    pdfp = '/home/ngr4/scratch60/scnd/data/processed/human_cellranger'
    ####
    
    for i, k in enumerate([3,5,10,15,30,100]):
        for j, n_pcs in enumerate([10,20,50,100]):
            if i+j==0:
                adata = init_load(pdfp)
            else:
                adata = load_pp_hum_adata()
            
            ####
            # plot names
            ####
            plot1 = os.path.join(pfp, 'hrbc_pre_k{}_npc{}.pdf'.format(k, n_pcs))
            plot2 = os.path.join(pfp, 'hrbc_post_k{}_npc{}.pdf'.format(k, n_pcs))
            plot3 = os.path.join(pfp, 'hrbc_violin_k{}_npc{}.pdf'.format(k, n_pcs))
            plot4 = os.path.join(pfp, 'hrbc_dot_k{}_npc{}.pdf'.format(k, n_pcs))
            plot5 = os.path.join(pfp, 'hrbc_umapcid_k{}_npc{}.pdf'.format(k, n_pcs))
            ####

            adata = scndpp.batch_correction(adata, 
                                            knn=k, n_pcs=n_pcs,
                                            plot1_out=plot1,
                                            plot2_out=plot2,
                                            plot3_out=plot3,
                                            plot4_out=plot4,
                                            plot5_out=plot5,)
            
            print('\n\nthrough {} trials in {:.1f}-min'.format(i+j+1, (time.time() - total)/60))
            del adata


