## import modules
import scanpy as sc
import pandas as pd
import seaborn as sns
import math
import random
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
from scipy.stats import pearsonr


import dcor
import pickle
from scipy.spatial.distance import euclidean

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report

from statsmodels.stats.multitest import multipletests


from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import ks_2samp
import statsmodels.stats.multitest as smm

## 
import numbers
import seaborn as sns
import networkx as nx

#test fucntion
def test():
    print('test')


## Differential features
def stripplot_from_adata(adata, groupby, feature, axs, medianline=False):
    """
    Generates a stripplot from an AnnData object, optionally overlaying a median line.
    
    Parameters:
    - adata: AnnData object containing scaled data and metadata in obs.
    - groupby: str, column name in adata.obs to group the data by.
    - feature: str, column name in adata.var or adata.to_df() to plot as y-axis.
    - axs: matplotlib.axes.Axes object to plot the stripplot on.
    - medianline: bool, whether to overlay the median line using a boxplot.
    
    Returns:
    - axs: matplotlib.axes.Axes object with the stripplot.
    """
    # Generate dataframe from adata
    data_df = adata.to_df(layer='raw')  # Assuming 'raw' layer exists, adjust if necessary
    
    # Append metadata
    data_df[['Gene', 'Media', 'Time']] = adata.obs[['Gene', 'Media', 'Time']].copy()
    
    # Create the stripplot
    sns.stripplot(
        data=data_df, 
        x=groupby, 
        y=feature, 
        ax=axs, 
        size=3, 
        color='gray', 
        alpha=0.75
    )
    
    # Add median line if requested
    if medianline:
        sns.boxplot(
            data=data_df, 
            x=groupby, 
            y=feature, 
            ax=axs, 
            showmeans=False, 
            meanline=False, 
            meanprops={'visible': False},  # Hide mean line
            medianprops={'color': 'red', 'linestyle': '-', 'linewidth': 1.5},  # Customize median line
            whiskerprops={'visible': False},  # Hide whiskers
            showfliers=False,  # Hide outliers
            showbox=False,  # Hide the box
            showcaps=False,  # Hide caps
            zorder=10  # Ensure it's above the stripplot
        )
    
    return axs

## Similarity Analysis

def pairwise_binary_clf(adata_in, gene1, gene2, use_pca):
    if use_pca: # if true, then use .obsm['X_pca']
        X = adata_in[adata_in.obs.Gene.isin([gene1, gene2])].obsm['X_pca']
    else:
        X = adata_in[adata_in.obs.Gene.isin([gene1, gene2])].X # **be careful whether you are using scaled vs raw. 
        
    y = adata_in[adata_in.obs.Gene.isin([gene1, gene2])].obs.Gene
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    model = LogisticRegression(solver='saga', max_iter = 1000, class_weight = 'balanced',
                               random_state = 0) # account for class imbalance
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test.values, y_pred) 
    # print(gene, 'non-target', auc)
    return auc

# create edges from AUC matrix
# df: pairwise comparisons showing AUC
def create_networkx_from_auc(df, thres, nodes):
    df = df.loc[nodes, nodes]
    df = df < thres # from AUC, smaller means "edge"
    node_labels = df.columns.tolist()
    df = df.astype('float').to_numpy()
    
    # Create a NetworkX graph from the adjacency matrix with node labels
    G = nx.from_numpy_matrix(df, create_using=nx.Graph())
    G = nx.relabel_nodes(G, {i: label for i, label in enumerate(node_labels)})
    
    return G

def jaccard_index_edges(G1, G2):
    # get jaccard index
    set1 = set(tuple(sorted(edge)) for edge in G1.edges())
    set2 = set(tuple(sorted(edge)) for edge in G2.edges())
    J = len(set1.intersection(set2))/len(set1.union(set2)) #
    return J

# shuffle and make edges
def permute_auc_dataframe(df, random_seed):
    #df=pairwise_clf_results_auc['D28_STD'].copy()
    vals = df.to_numpy()
    nodes_to_shuffle = df.columns.tolist()
    random.seed(random_seed)
    random.shuffle(nodes_to_shuffle)
    shuffled_df = pd.DataFrame(index = nodes_to_shuffle, columns = nodes_to_shuffle, data = vals)
    return shuffled_df

def degree_preserving_ppi_shuffling(G_in, rand_seed):
    deg_seq = [d for n,d in G_in.degree()] # construct degree info
    mapping = dict(zip(range(len(G_in.nodes())), G_in.nodes()))
    G_rand = nx.random_degree_sequence_graph(deg_seq, seed=rand_seed)
    G_rand = nx.relabel_nodes(G_rand, mapping)
    return G_rand

# jaccard index of "significant features", but signed
def similarity_score(ks_results, gene1, gene2, thres):
    df1 = ks_results[gene1]
    df2 = ks_results[gene2]
    # subset significant ones based on defined threshold. exmaple. = 0.05 padjsted
    df1 = df1[df1.p_adj<thres]
    df2 = df2[df2.p_adj<thres]
    # calculate jaccard index - for each sign, adapted from Tian et al 2019
    # positive
    pos_df1 = df1[df1.Median_Z > 0]
    pos_df2 = df2[df2.Median_Z > 0]
    
    # negative
    neg_df1 = df1[df1.Median_Z < 0]
    neg_df2 = df2[df2.Median_Z < 0]
    
    pos_intersection = len(set(pos_df1.index).intersection(set(pos_df2.index)))
    pos_union = len(set(pos_df1.index).union(set(pos_df2.index)))
    
    neg_intersection = len(set(neg_df1.index).intersection(set(neg_df2.index)))
    neg_union = len(set(neg_df1.index).union(set(neg_df2.index)))
    
    S = (pos_intersection + neg_intersection)/(pos_union + neg_union)
    return S


### Composition analysis
def hypergeom_custom(adata, group1, group2):
    #from scipy.stats import hypergeom
    # group1 is "GO term", group 2 is "DEG" as an easy analogy
    group1_list = sorted(list(set(adata.obs[group1]))) # Y
    group2_list = sorted(list(set(adata.obs[group2]))) # X
    if 'non-target' in group2_list:
        group2_list.remove('non-target')
        group2_list = ['non-target'] + group2_list
    # initialize
    pval_df = pd.DataFrame(index=group1_list, columns = group2_list)
    proportion_df = pd.DataFrame(index=group1_list, columns = group2_list)
    
    for grp1 in group1_list:
        for grp2 in group2_list:
        # get M, n, N, x
            M = adata.shape[0] # total number of objects --> total cells
            n = adata[adata.obs[group1]==grp1].shape[0] # --> type I object, cluster
            N = adata[adata.obs[group2] == grp2].shape[0]  # 
            x = adata[(adata.obs[group1]==grp1)&(adata.obs[group2] == grp2)].shape[0]
            # --> number of type I objects in N drawn
            p_value = 1 - stats.hypergeom.cdf(x, M, n, N)
            pval_df.loc[grp1, grp2] = p_value
            proportion_df.loc[grp1, grp2] = x/N
    return proportion_df, pval_df

# create adjusted pvalue
def correct_pvals(pval_df, alpha=0.05, method='fdr_bh'):
    from statsmodels.stats.multitest import multipletests
    pval_flattened = pval_df.values.reshape(-1)
    
    # perform multiple hypothesis correction
    _, corrected_pvals, _, _ = multipletests(pval_flattened, alpha=alpha, method = method)
    adj_pval_df = pd.DataFrame(corrected_pvals.reshape((pval_df.shape)), 
                               index = pval_df.index, columns = pval_df.columns)
    return adj_pval_df



### plot
# utility function for differential feature analysis
def plot_cdf(array1, array2, ax=None, label1='Array A', label2='Array B', title='CDF Comparison', xlabel='Values', ylabel='CDF'):
    """
    Plots the cumulative density functions (CDF) of two arrays.

    Parameters:
        array1 (array-like): First array of values.
        array2 (array-like): Second array of values.
        label1 (str): Label for the first array.
        label2 (str): Label for the second array.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.

    Returns:
        None
    """
    # Get current axes if none are provided
    if ax is None:
        ax = plt.gca()
    
    # Compute sorted values and their CDF for the first array
    sorted_a = np.sort(array1)
    cdf_a = np.arange(1, len(sorted_a) + 1) / len(sorted_a)

    # Compute sorted values and their CDF for the second array
    sorted_b = np.sort(array2)
    cdf_b = np.arange(1, len(sorted_b) + 1) / len(sorted_b)

    # Plot the CDFs
    plt.figure(figsize=(8, 6))
    ax.plot(sorted_a, cdf_a, label=label1, color = 'blue', linestyle='-' )
    ax.plot(sorted_b, cdf_b, label=label2, color = 'orange',  linestyle='-')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    ax.set_facecolor('white')
    
    ax.spines['top'].set_visible('False')
    ax.spines['right'].set_visible('False')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    
    #ax.legend()
    legend = ax.legend()
    legend.get_frame().set_facecolor('white') 

# utility function for differential feature analysis
def annotate_overlapping_features(df1, df2, Z_thres, pval_thres, center_val):
    features1 = [x for x in df1[(abs(df1[center_val])>Z_thres)&(df1.p_adj<pval_thres)].index]
    features2 = [x for x in df2[(abs(df2[center_val])>Z_thres)&(df2.p_adj<pval_thres)].index]
    inds = set(features1).intersection(features2)
    # initialize. 
    df1.loc[:, 'overlap'] = 0
    df2.loc[:, 'overlap'] = 0
    
    df1.loc[inds, 'overlap'] =1
    df2.loc[inds, 'overlap'] =1
    return list(inds), df1, df2 

def annotate_protein_volcano(df, Z_thres, pval_thres, protein, veto, center_val, categ = None, R2_thres = 1.0):
    features = [x for x in df[(abs(df[center_val])>Z_thres)&(df.p_adj<pval_thres)&(df.R2_QQplot < R2_thres)].index]
    annotated = []
    for x in features:
        if protein in x.split('_'):
            if veto is None:
                annotated.append(x)
            elif categ is None:
                if veto not in x.split('_'):
                    annotated.append(x)
            
            else:
                if (veto not in x.split('_'))&(categ in x.split('_')):
                    annotated.append(x)
                
                
#    annotated = [x for x in features if protein in x.split('_')]
    
    # initialize. 
    df.loc[:, 'annotate'] = 0
    df.loc[annotated, 'annotate'] =1
    return annotated, df


def plot_diff_features(df, ax, hue, xaxis, yaxis, color_map, gene, **kwargs):
    """
    Flexible scatter plot function to visualize data with customizable styling.

    Parameters:
    - df: DataFrame containing the data.
    - ax: Axes object for the plot.
    - hue: Column name in df for color coding.
    - xaxis: Column name in df for the x-axis.
    - yaxis: Column name in df for the y-axis.
    - color_map: Dictionary mapping hue levels to colors.
    - gene: Title of the plot.
    - **kwargs: Additional arguments for customization.
        - s: Marker size.
        - linewidth: Spine line width.
        - xlabel_fontsize: Font size for the x-axis label.
        - ylabel_fontsize: Font size for the y-axis label.
        - title_fontsize: Font size for the title.
        - facecolor: Background color of the plot.
        - spines_color: Color of the spines.
    """
    # Scatter plot
    sns.scatterplot(
        data=df, 
        x=xaxis, 
        y=yaxis, 
        hue=hue, 
        palette=color_map, 
        s=kwargs.get('s', 15),  # Marker size, default is 15
        ax=ax, 
        legend=kwargs.get('legend', False)  # Default: no legend
    )
    
    # Customize the appearance
    ax.set_facecolor(kwargs.get('facecolor', 'white'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    spine_color = kwargs.get('spines_color', 'black')
    linewidth = kwargs.get('linewidth', 2.5)
    ax.spines['bottom'].set_color(spine_color)
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    
    # Customize labels and title
    ax.set_xlabel(xaxis, fontsize=kwargs.get('xlabel_fontsize', 20))
    ax.set_ylabel(yaxis, fontsize=kwargs.get('ylabel_fontsize', 20))
    ax.set_title(gene, fontsize=kwargs.get('title_fontsize', 25))
    ax.tick_params(axis='x', labelsize=kwargs.get('xtick_fontsize', 15))  # X-axis tick label font size
    ax.tick_params(axis='y', labelsize=kwargs.get('ytick_fontsize', 15))  # Y-axis tick label font size

def diff_feature_distribution(data1, data2, gene1, gene2, feature, ax, **kwargs):
    # kwargs; xaxis_label, yaxis_label, facecolor, spines_color, xtick_fontsize, ytick_fontsize, 
    
    n_bins = round(np.sqrt(data1.shape[0]))+round(np.sqrt(data2.shape[0]))
    MAX = np.max([data1.max(), data2.max()])
    MIN = np.min([data1.min(), data2.min()])
    
    
    sns.histplot(data = data1, kde=True, color = 'blue', alpha = kwargs.get('alpha', 0.25), ax = ax, 
                 legend=kwargs.get('legend', False), stat='proportion', 
                 bins = n_bins, binrange = (MIN, MAX))
    sns.histplot(data = data2, kde=True, color = 'orange', alpha = kwargs.get('alpha', 0.25), ax = ax, 
                 legend=kwargs.get('legend', False), stat='proportion', 
                 bins =n_bins, binrange = (MIN, MAX))
    
    ax.set_facecolor(kwargs.get('facecolor', 'white'))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    spine_color = kwargs.get('spines_color', 'black')
    linewidth = kwargs.get('linewidth', 2.5)
    ax.spines['bottom'].set_color(spine_color)
    ax.spines['left'].set_color(spine_color)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.set_xlabel(kwargs.get('xaxis_label', 'Z-score'), fontsize=kwargs.get('xlabel_fontsize', 20))
    ax.set_ylabel(kwargs.get('yaxis_label', 'Proportion'), fontsize=kwargs.get('ylabel_fontsize', 20))
    ax.tick_params(axis='x', labelsize=kwargs.get('xtick_fontsize', 15))  # X-axis tick label font size
    ax.tick_params(axis='y', labelsize=kwargs.get('ytick_fontsize', 15))  # Y-axis tick label font size
    
    
    
### Utility functions for plotting example single-cells    

def get_cells(grp1, grp2, n_cells, seed=None):
    """
    Randomly samples n_cells from two cell groups (grp1 and grp2).

    Parameters:
    grp1 : AnnData object
        First group of cells.
    grp2 : AnnData object
        Second group of cells.
    n_cells : int
        Number of cells to sample from each group.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    cells1 : list
        List of sampled cell indices from grp1.
    cells2 : list
        List of sampled cell indices from grp2.
    """
    # Apply random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    # Get cell indices from the observation index
    _cells1 = list(grp1.obs.index)
    _cells2 = list(grp2.obs.index)
    
    # Check if the number of cells to sample is valid
    if n_cells > len(_cells1) or n_cells > len(_cells2):
        raise ValueError("n_cells cannot be greater than the number of cells in either group")
    
    # Randomly sample n_cells from each group
    cells1 = random.sample(_cells1, n_cells)
    cells2 = random.sample(_cells2, n_cells)
    
    return cells1, cells2

def subset_adata_by_feature_ranges(adata, feature_name, MIN, MAX):
    
    # Identify the index of the feature in adata.var
    if feature_name in adata.var.index:
        feature_idx = adata.var.index.get_loc(feature_name)
    else:
        raise KeyError(f"Feature '{feature_name}' not found in adata.var.")
    
    # Subset cells where the feature's values are between 0 and 1
    mask = (adata.X[:, feature_idx] >= MIN) & (adata.X[:, feature_idx] <= MAX)
    subset_adata = adata[mask, :]
    
    return subset_adata

def plot_cells(data, cell_id, ax, vmin, vmax):
    images = data.loc[cell_id]
    
    
    
    images1 = im_df.loc[cells1]
    images2 = im_df.loc[cells2]
    # plot each cells
    for i in range(len(cells1)):
        cell1 = cells1[i]
        cell2 = cells2[i]
        axs[0][i].imshow(images1.loc[cell1].image[ch,...])
        axs[1][i].imshow(images2.loc[cell2].image[ch,...])
    

    