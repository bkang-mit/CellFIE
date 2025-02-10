```1_Prepare_Prize_Inputs_and_Interactome.ipynb```: Prepares input prize file for OmicsIntegrator, including parsing from HD genetics and Drosophila data, calculating cell type enrichment score. This code also generates input interactome file with cell type specificity score. 

```2_cortex_celltype_specific_OmicsIntegrator_GridSearch.py```: Takes input prizes and interactome file to generate cell type naive and specific networks for each cell type. It performs grid search over selected hyperparameter ranges and calculates various metrics for final network selection. ```graph_cell_network_edge_update.py``` provides updated wrapper functions to take additional hyperparameters related to cell type specific adjustments (adapted from OmicsIntegrator 2 [package](https://github.com/fraenkel-lab/OmicsIntegrator2))

```3_1_Inspect_and_Filter_CellSpecificNetworks.ipynb```: Load all ranodmization grid search results and select optimal hyperparameter. For each cell type network, input files for GSEA (gene sets) are generated. Cell type specificity metrics are evaluated with outputs of GSEA (see R script in part 3_2 and 3_3 below), related to Figure 2B

```3_2_FGSEA_R.ipynb``` ```3_3_FGSEA_R_PrizeSet.ipynb```: R scripts used to perform GSEA using fGSEA package. Related to Figure 2B, S1

```4_Clustering_pathway_enrichment_Network_visualization.ipynb```: Performs Louvain Clustering to obtain subnetworks and gProfiler enrichment analysis to annotate pathways. Related to Figures 2C-E
