'''


Samples:
    CTRL- 15162, 4092
    SCA1- 122, 132
    SCA3- 32, 33, 34, 35
    
'''

import os
import time
import datetime
import glob
import scanpy as sc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scprep
import graphtools as gt
import phate
from scipy import sparse
from scipy.stats import zscore 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.stats import mannwhitneyu, tiecorrect, rankdata
from statsmodels.stats.multitest import multipletests
import warnings
import sys
sys.path.append('/home/ngr4/project/scripts/')
import utils

plt.rc('font', size = 8)
plt.rc('font', family='sans serif')
plt.rcParams['pdf.fonttype']=42
plt.rcParams['ps.fonttype']=42
plt.rcParams['legend.frameon']=False
sns.set_style("ticks")

dfp1 = '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/022520/*/filtered_feature_bc_matrix/'
dfp2 = '/gpfs/ycga/sequencers/pacbio/gw92/10x/data_sent/llt26/070920/*/filtered_feature_bc_matrix/'
data_files = glob.glob(dfp1) + glob.glob(dfp2)
sample_suffix = '_HNT_cellranger'

pfp = '/home/ngr4/project/scnd/results'
pdfp = '/home/ngr4/project/scnd/data/processed/'

if True:
    # first load
    adatas = {}
    for i, file in enumerate(data_files):
        if i==0:
            adata = sc.read_10x_mtx(file)
            batch_key = file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]
            adata.var_names_make_unique()
        else:
            adatas[file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]] = sc.read_10x_mtx(file)
            adatas[file.split('/filtered_')[0].split('/')[-1].split(sample_suffix)[0]].var_names_make_unique()

    adata = adata.concatenate(*adatas.values(), batch_categories=[batch_key]+list(adatas.keys()))
    del adatas
    
# filter cells/genes, transform
sc.pp.calculate_qc_metrics(adata,inplace=True)
mito_genes = adata.var_names.str.startswith('MT-')
adata.obs['pmito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
print('Ncells=%d have >10percent mt expression' % np.sum(adata.obs['pmito']>0.1))
print('Ncells=%d have <200 genes expressed' % np.sum(adata.obs['n_genes_by_counts']<200))
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3) # filtering cells gets rid of some genes of interest
adata.raw = adata
adata = adata[adata.obs.pmito <= 0.1, :]
sc.pp.normalize_total(adata)
sc.pp.sqrt(adata)
sc.tl.pca(adata, n_comps=100)

if True :
    # save
    adata.write(os.path.join(pdfp,'hum_wsca3.h5ad'))
    print('\n... saved @'+datetime.datetime.now().strftime('%y%m%d.%H:%M:%S'))
    
markers = {
    'Granule cell':['GABRA6','SLC17A7'],
    'Unipolar brush cell':['SLC17A6','EOMES'],
    'Purkinje cell':['ATP2A3','CALB1','CA8','PPP1R17','SLC1A6'],
    'Inhibitory interneuron':['GAD1','GAD2','NTN1','MEGF10'],
    'Astrocyte':['ALDH1L1','AQP4'],
    'Bergmann glia':['GDF10','HOPX'],
    'OPC':['OLIG1','OLIG2','PDGFRA'],
    'OL':['HAPLN2','MAG','MOG','OPALIN'],
    'Microglia':['C1QB','CX3CR1','DOCK2','P2RY12'],
    'Pericytes':['FLT1','RGS5'],
    'Endothelial cell':['DCN','LUM','KDR'],
}

