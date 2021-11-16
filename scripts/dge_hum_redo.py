import sys
sys.path.append('/home/ngr4/project/')
from scnd.scripts import dge_scnd as scnddge
from scnd.scripts import utils as scndutils


import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

import scanpy as sc
import bbknn
import phate

def dge_per_group(adata, up_grp, down_grp, group='ctype'):
    '''
    Arguments:
      up_grp (str): key for positive EMD values indicating up in this group
      down_grp (str): key for negative EMD values indicating down in this group
      group (str): key for the sc.AnnData.obs slot, which is a pd.DataFrame  
    '''
    

def dge_hum():
    adata = scnddata.load_annotated_hum_redo()
    
    
if __name__ == '__main__':
    dge_hum()

