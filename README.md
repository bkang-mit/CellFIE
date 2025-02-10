# CellFIE: Cell-intrinsic pathway discovery framework with Functional, In-situ profiling Experiments

### Overview
This repo contains codes used for CellFIE manuscript. "src_network_analysis" folder contains codes used to generate networks and their related analysis and "src_image_analysis" folder contains codes related to image preprocessing and analysis

### Conda Environment Setup
1. src_network_analysis: create and activate conda environment with networks.yml
2. src_image_analysis: create and activate conda environment with imlab2.yml
3. image_preprocessing: follow a separate repo linked [here](https://github.com/fraenkel-lab/image-preprocessing) to perform image-preprocessing (Registration & Cell Pose segmentation)

### Section 1: Network Analysis
This section provides codes for 1) Parsing and Preprocessing Prize and Interactome 2) Cell type specific weight adjustment and Omics Integrator Grid Search 3) Post-processing and filtering of network results 4) Clustering and pathway enrichment analysis
Detailed description of each code is provided in src_network_analysis/README.md

### Section 2: Image Analysis
This section provides codes for 1) Matching Pursuit Deconvolution 2) Cell Profiler Feature Extraction 3) Barcode Calling with QC 4) Single-cell morphological fingerprint analysis and 5) neuronal arborization analysis.
Detailed description of each code is provided in src_image_analysis/README.md

