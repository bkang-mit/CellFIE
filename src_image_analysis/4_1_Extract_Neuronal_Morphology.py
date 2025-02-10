########### import modules #####################
import scanpy as sc
import pandas as pd
import seaborn as sns
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import os
import sys
import glob
from imageio import volread as imread
from skimage.filters import threshold_otsu

from skimage import measure
from scipy import stats
import umap
from sklearn.decomposition import PCA
import math
from scipy import stats

import tifffile
import networkx as nx
import pickle
# computer vision related modules
# or https://github.com/Image-Py/sknw
import sknw
from skimage.filters import meijering, sato, frangi, hessian
from skimage.morphology import skeletonize, square, closing
#from skan import skeleton_to_csgraph
from skimage.draw import circle_perimeter
from skimage.segmentation import find_boundaries



# load metadata - load full codebook as well
META_DIR = 'metadata'
full_codebook = pd.read_csv(f'../{META_DIR}/full_codebook.csv',sep=',', index_col=0) # this is "legal" codebook
Procode_gRNA = pd.read_csv(f'../{META_DIR}/PROCODE_gRNA.csv',sep=',')
legal_codes = sorted(list(set(Procode_gRNA['ProCode ID'].to_list())))
codebook = full_codebook[legal_codes]
#AllProcodes = pd.read_csv('AllProcodes.csv', sep='.')
markers = pd.read_csv(f'../{META_DIR}/clean_markers.csv')

## Assign Gene ID
procode_gene_df = pd.read_csv('../metadata/PROCODE_gRNA.csv')
procode_to_gene = {}
for cell in procode_gene_df.index:
    procode = procode_gene_df.loc[cell,'ProCode ID']
    if procode not in procode_to_gene.keys():
        procode_to_gene[procode] = procode_gene_df.loc[cell,'Gene Target']

######## Useful Functions ####################3
def get_cell_coords(cell_label, coord_data):
    
    x2 = coord_data.loc[cell_label, 'AreaShape_BoundingBoxMaximum_X']
    y2 = coord_data.loc[cell_label, 'AreaShape_BoundingBoxMaximum_Y']
    x1 = coord_data.loc[cell_label, 'AreaShape_BoundingBoxMinimum_X']
    y1 = coord_data.loc[cell_label, 'AreaShape_BoundingBoxMinimum_Y']
    
    return round(x1),round(x2),round(y1),round(y2)

def get_edge_length(g):
    w = [g[s][e]['weight'] for (s,e) in g.edges()] #list of all edge length in graph g
    return sum(w)

def get_subgraph_stats(img, connected_subgraphs):
    df = pd.DataFrame()
    k=0
    for i in range(img.shape[0]):
        subgraphs = connected_subgraphs[i]  # subgraph for each channel
        for j in range(len(subgraphs)): # for each subgraph
            g = subgraphs[j]
            df.loc[k,'Number_of_Nodes'] = g.number_of_nodes()
            df.loc[k, 'Number_of_Edges'] = g.number_of_edges()
            df.loc[k,'Subgraph_Index'] = j
            df.loc[k, 'Channel_Index'] = i
            
            Edge_length = get_edge_length(g)
            df.loc[k, 'Edge_Length'] = Edge_length
            nodes = g.nodes()
            degrees = [g.degree[n] for n in nodes]
            avg_degree = np.mean(degrees)
            df.loc[k, 'Mean_Degree'] = avg_degree
            
            branch_nodes = [n for n in nodes if g.degree[n]>2] # list of nodes that have degree>2
            df.loc[k, 'Number_of_Branches'] = len(branch_nodes) 
            k = k+1 # update index
    return df

def convert_binary(img, close_size): # returns both raw_binary and closed_binary
    # inputs: 
    # img: 2D maximum projected, MP_SCORE_MAX images (CYX)
    # close_size = 5 for closing operation.
    close_structure = square(close_size)
    img_bin = np.zeros(img.shape)
    img_bin_clo = np.zeros(img.shape)
    for ch in range(img.shape[0]):
        img_bin[ch] = np.where(img.max(0)==0, 0, img.argmax(0)==ch)
        img_bin_clo[ch] = closing(img_bin[ch], footprint=close_structure) # close out gaps 
    return img_bin, img_bin_clo 

def extract_features(allFOVs, DATA_DIR, SEG_DIR, adata):
    graph_store = {} # nested dictionary 
    subgraph_stats_store = {} # each fov, store dataframe of subgraph statistics.
    obs_df = adata.obs.copy() # record Trunks, Sholl_1
    obs_df['UMI'] = adata.obs.index.tolist() # backup 
    
    for ii in range(len(allFOVs)): # for each image
        fov = allFOVs[ii]
        
        im_id = f'F{fov}_max.tif'
        cell_df = adata.obs[adata.obs.FileName_max == im_id]
        soma = imread(f'../{DATA_DIR}/{SEG_DIR}/F{fov}_max_Soma.tiff')
        print('loading input mp_score_max', f'F{fov}_max_score_max.tif')
        img = imread(f'../{DATA_DIR}/{MP_DIR}/F{fov}_mp_score_max.tif')
        # make binary, closed image with varying scales. - from previous analysis, square(15) seems fine.
        #_, img_bin_5 = convert_binary(img, 5)
        _, img_bin_15 = convert_binary(img, 5) # USE pixel size = 5 for closing. bin_15 is just naming. no meaning
        
        # skeletonize closed binary image - bug: instead of 1 and 0 this returned 255
        bin_skel_15 = skeletonize(img_bin_15).astype('bool')
        bin_skel_15 = bin_skel_15.astype('float')
        if bin_skel_15.max() > 1:
            print('check bin_skel_15 data type')
            
        
        if os.path.exists(f'../{DATA_DIR}/mask/F{fov}_mask.tif'):
            box_mask = imread(f'../{DATA_DIR}/mask/F{fov}_mask.tif').astype('bool')      
        #masked_skel_5 = np.zeros(bin_skel_5.shape)
        masked_skel_15 = np.zeros(bin_skel_15.shape) # initialize
        
        for ch in range(bin_skel_15.shape[0]):
            coord_data = cell_df[cell_df.Barcode_Idx == f'intensity_mean-{ch}']
            cells = coord_data.ObjectNumber.values.tolist()
            mask = np.zeros(soma.shape)
            for cell in cells: # for all cells in coord_data
                mask = mask + (soma==cell) # append masks for all cells in each channel.
                
                # we only want to mask out soma that belongs to each "Barcode"
                # for example we don't want barcode 1 soma to mask out barcode 2 neurite.
                # we just want to mask out barcode 1 neurite with barcode 1 soma masked out.
            if os.path.exists(f'../{DATA_DIR}/mask/F{fov}_mask.tif'):
                    #masked_skel_5[ch] = bin_skel_5[ch]*(~mask.astype('bool'))*(~box_mask) # mask out soma, mask out bbox
                masked_skel_15[ch] = bin_skel_15[ch]*(~mask.astype('bool'))*(~box_mask) # mask out soma, mask out bbox
            else:
                #masked_skel_5[ch] = bin_skel_5[ch]*(~mask.astype('bool')) # just mask out soma 
                masked_skel_15[ch] = bin_skel_15[ch]*(~mask.astype('bool'))
            
            ##################################################################
            umi = coord_data.index.tolist() # list of unique identifiers(**BEFORE RESET OF INDEX!)
            coord_data = coord_data.set_index('ObjectNumber') # for each channel
            coord_data['umi'] = umi # record unique identifier
            
            # Trunk gets un-masked because we need to calculate intersection with soma boundary. 
            #skel_im_5 = bin_skel_5[ch]
            skel_im_15 = bin_skel_15[ch]
            
            # Calculate Trunk for each cell
            for cell in cells:
                x1,x2,y1,y2 = get_cell_coords(cell,coord_data) # get coordinate of each cell
                center = (round((y2-y1)/2), round((x2-x1)/2)) # find center point
                radius = round(np.sqrt(center[0]**2 + center[1]**2)) # define radius of initial sholl 1
                R = round(np.sqrt(center[0]**2 + center[1]**2)) # define radius of initial sholl 1
                R1 = 0 # level 1, slightly around bounding box
                rr1, cc1 = circle_perimeter(y1+center[0], x1+center[1], R+R1) # sholl level 1
                _bound = (soma == cell) # roi
                bound = skeletonize(find_boundaries(_bound)).astype('uint16') # 
                
                inv_soma = ~(soma==cell)
                cy, cx = center[0]+y1, center[1]+x1
                
                # check out of bounds
                if (cy-1-R < 0)|(cx-1-R < 0) | ((cy+1+R) > (skel_im_15.shape[0]-1))|((cx+1+R)>(skel_im_15.shape[1]-1)):
                    print('out of bounds')
                    umi = coord_data.loc[cell, 'UMI']
                    
                    # CAREFUL - obs_df has 'string' indices; umi is float
                    obs_df.loc[str(umi), 'Trunk_ID'] = cell
                    obs_df.loc[str(umi), 'Trunks'] = np.nan
                    obs_df.loc[str(umi), 'Sholl_1'] = np.nan
                
                else:
                    circle1 = np.zeros(inv_soma.shape) # initialize circle
                    circle1[rr1,cc1] = 1 # define sholl 1 circle
                    
                    inv_soma = inv_soma[cy-1-R:cy+1+R, cx-1-R:cx+1+R] # 
                    intersection1 = skel_im_15[cy-1-R:cy+1+R, cx-1-R:cx+1+R]*inv_soma * circle1[cy-1-R:cy+1+R, cx-1-R:cx+1+R]
                    intersection_bound=skel_im_15[cy-1-R:cy+1+R, cx-1-R:cx+1+R]*bound[cy-1-R:cy+1+R, cx-1-R:cx+1+R]
                    umi = coord_data.loc[cell,'UMI']
                    obs_df.loc[str(umi), 'Trunk_ID'] = cell
                    obs_df.loc[str(umi), 'Trunks'] = np.sum(intersection_bound)
                    obs_df.loc[str(umi), 'Sholl_1'] = np.sum(intersection1)
                    
        # initialize dictionary to make graph for each channel
        ## For each fov, save graph_store_fov, subgraph_stats_store
        graphs = {}
        for i in range(masked_skel_15.shape[0]):
            graphs[i] = sknw.build_sknw(masked_skel_15[i]) # for each channel, make master graph
        
        connected_subgraphs = {}
        for i in range(masked_skel_15.shape[0]): # list of subgraphs
            connected_subgraphs[i] = [graphs[i].subgraph(c) for c in nx.connected_components(graphs[i])]
        # graph_store[fov] = connected_subgraphs # store entire dictionary of connected subgraphs - for later op
        # connected_subgraphs[0][1] is 0th channel, 1th subgraph
        
        
        if not os.path.exists(f'../{DATA_DIR}/morphology_data'):
            os.mkdir(f'../{DATA_DIR}/morphology_data')
        
        # save connected_subgraphs (graph_store[fov] formerly)
        with open(f'../{DATA_DIR}/morphology_data/graph_store_{fov}.pkl', 'wb') as handle:
            pickle.dump(connected_subgraphs, handle)
        
        stat_df=get_subgraph_stats(masked_skel_15, connected_subgraphs)
        #subgraph_stats_store[fov] = stat_df

        # save connected_subgraphs
        with open(f'../{DATA_DIR}/morphology_data/subgraph_stats_store_{fov}.pkl', 'wb') as handle:
            pickle.dump(stat_df, handle)
            
    return graph_store, subgraph_stats_store, obs_df


# Load cell level data and run program
DATA_DIRs = ['101222_D10_Coverslip1_Processed', '102222_D10_Coverslip4_Reimage_Processed',
             '102422_D21_Coverslip7_Processed', '103122_D28_Coverslip13_Processed',
             '110622_D28_Coverslip16_Processed']
print(DATA_DIRs)             

for DATA_DIR in DATA_DIRs:
    print('===============================================================================')
    print("Processing...", DATA_DIR)
    
    # Load Feature Data
    MP_DIR = 'mp_score_max'
    SEG_DIR = 'Cellpose_Segmentation_masks'
    DATA_TYPE = 'max'
    META_DIR = 'metadata'
    
    _allFOVs = sorted(glob.glob(f'../{DATA_DIR}/{MP_DIR}/*'))
    allFOVs = [x.split('F')[-1][:3] for x in _allFOVs]
    print(len(allFOVs))
    
    ADATA_DIR = 'adata_Cellpose'
    SAMPLE = DATA_DIR.split('_')[2]
    
    adata = sc.read(f'../{DATA_DIR}/{ADATA_DIR}/{SAMPLE}_adata_Cellpose.h5ad')
    print(adata.obs.PathName_max)  
    
    graph_store, subgraph_stats_store, obs_df = extract_features(allFOVs, DATA_DIR, SEG_DIR, adata)
    
    if not os.path.exists(f'../{DATA_DIR}/morphology_data'):
        os.mkdir(f'../{DATA_DIR}/morphology_data')
    
    # For memory issue, save each fov at a time.
    #with open(f'../{DATA_DIR}/morphology_data/subgraph_stats_store.pkl', 'wb') as handle:
    #    pickle.dump(subgraph_stats_store, handle)
    
    #with open(f'../{DATA_DIR}/morphology_data/graph_store.pkl', 'wb') as handle:
    #    pickle.dump(graph_store, handle)
    
    # trunk can stay as it is.
    with open(f'../{DATA_DIR}/morphology_data/morphology_data.pkl', 'wb') as handle:
        pickle.dump(obs_df, handle)
