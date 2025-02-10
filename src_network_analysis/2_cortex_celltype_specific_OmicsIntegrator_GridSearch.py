import os
#os.chdir('/nfs/latdata/bkang/Codes_Publication_2024')
#print(os.getcwd())
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import OmicsIntegrator as oi
import warnings
#warnings.filterwarnings('ignore')
# Plotting
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "Arial"

import seaborn as sns

###################################
# modified omics integrator package for incorporating cell type specific adjustments.
import graph_cell_network_edge_update as coi
####################################
print(os.getcwd())
interactome_file = '../Interactome/iref17_iref14_braInMap_ptau_metab_mapped_cortex_adjusted.txt'
print(interactome_file)

# hyperparam choices.
Ws = [1]
Bs = [50, 100]
Gs = [4,5]
Ks = [5, 10, 15]

params = {
    "noise": 0.1, # gaussian noise to randommization
    "dummy_mode": "terminals", # connect dummy to terminals
    "exclude_terminals": False, 
    "seed": 1
}
prize_file_prefix = 'prize_cortex_10_'
# run for each cell type - for each cell type, run 100 randomizations to assess robustness/specificity.
celltypes = ['Ex','In','Ast', 'Oli','Opc','Mic']
for cell in celltypes:
    prize_file = '../Prize_Inputs/' + prize_file_prefix + cell + '.txt'
    print(prize_file)
    params['cell_type'] = cell
    print('Cell Network: ', params['cell_type'])

    graph = coi.Graph(interactome_file, params)   
    robustness_reps = 100
    specificity_reps = 100
    #cell_specificity_reps = 100
    
    print(robustness_reps, specificity_reps)
    cell_type = cell
    print(cell_type, 'running grid rand')
    results = graph.grid_randomization(prize_file, cell_type, Ws, Bs, Gs, Ks, robustness_reps, specificity_reps)
    filename="{}_cortex_w10_randomization_results.pkl".format(cell)
    output_dir = "../Network_Outputs/pickles/111221_cortex/" + filename
    with open(output_dir, "wb") as f: # change directory to output
        pickle.dump(results, f)
    print(filename, 'saved')