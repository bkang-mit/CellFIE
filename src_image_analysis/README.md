# Scripts for generating main and supplementary figures of Kang et al (in prep)

### Part 1: Matching Pursuit Deconvolution and Cell Profiler Feature extraction
```1_Prepare_Inputs_CP_MP-gcloud.py``` : Code for generating Matching Pursuit (MP) ouptut files and Cell Profiler input files (maximum-intensity-projected images). Processed in gcloud. Example notebook with a demo data is shown in ```1_Prepare_Inputs_CP_MP-Example.ipynb```

```1_2_cp_analysis_cellpose_gcloud.cppipe``` and ```1_3_cp_analysis_cellpose_posthoc.cpproj``` : Cell Profiler pipelines to extract single-cell image features. Followed this [instruction](https://carpenter-singh-lab.broadinstitute.org/blog/getting-started-using-cellprofiler-command-line) install CellProfiler package and to run pipeline headlessly

### Part 2: Debarcoding and Quality Control to assign barcodes to single-cell
```2_1_Debarcoding_Soma_Generate_QC_File.ipynb```: Takes the Matching Pursuit Deconvolved image and outputs various QC metrics for post-processing of barcoding results. Example notebook with a demo data is shown in ```2_1_Debarcoding_Soma_Generate_QC_File-Example.ipynb```
```2_2_Debarcoding_Soma_Barcode_Calling.ipynb```: Notebook demonstrating false assignment classifier - Figure S4
```2_3_Combine_CellProfiler_and_Barcode_Assignment.ipynb```: Generates intermediate file that combines phenotypes and genotypes

### Part 3: Morphological Fingerprint and Differential morphology analysis
```3_1_Create_AnnData_Single_Cell_Features.ipynb```: Creates AnnData Object with all metadata combined. 
```3_2_Morphological_Fingerprint_Analysis-Fig4.ipynb```: Performs analysis workflow, related to Figure 4, S5-7

### Part 4: Neuronal arborization analysis
```4_1_Extract_Neuronal_Morphology.py```: Extracts arborization parameters from Matching Pursuit Deconvolved images. Intermediate outputs are generated, including networkx graph objects and cells-by-num_trunks dataframe
```4_2_Neurite_Morphology_Analysis-Fig5.ipynb```: Compile all feature intermediate outputs and Construct replicate-by-feature data matrix normalizing each replicate by number of cells (neurite length, num_branches). Contains all analysis performed for Figure 5, Figure S8-9

