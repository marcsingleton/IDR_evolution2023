# IDR Evolution

This is the repo for the IDR Evolution project. Its aim is to identify evolutionarily conserved properties of intrinsically disordered regions (IDRs) in proteins. Ideally IDRs across different proteins would share similar patterns of conserved properties, allowing them to be grouped into classes similar to the classification systems used for structured domains, such as Pfam. These analyses focus on a set of orthologous groups of proteins found in 33 genomes within the genus *Drosophila*, which were identified in the companion project [Orthology Inference](https://github.com/marcsingleton/orthology_inference2023). These methods and their results are discussed in further detail in the accompanying [pre-print](https://www.biorxiv.org/content/10.1101/2023.12.05.570250v1).

## Project Organization

At the highest level, this project is organized into the following components:

```
IDR_evolution/
	├── analysis/
	├── bin/
	├── data/
	├── src/
	└── README.md
```

Only `analysis/` and `src/`, which together contain all code written for this project, are explicitly tracked by Git. `bin/` contains third-party programs or code used in this project. Though this directory is not tracked by Git, scripts may reference it by path, so it is included here for completeness. Similarly, `data/`, which contains all the raw data used in this project, is not tracked by Git.

`analysis/` contains only directories, which serve to group related analyses. Some directories are "orphaned" and no longer contribute to any recent or ongoing analyses, but are included here for completeness. Currently, it contains the following entries:
- `brownian/`: Application of Brownian motion model to orthologs from NCBI annotations, among other phylogenetic analyses
- `evofit/`: Fitting various phylogenetic substitution models to alignments
- `GO/`: Analyses using GO terms and the results of other analyses
- `IDRpred/`: Prediction of IDRs in alignments and other analyses used to partition the data
- `TFCF/`: Analyses to parse and deduplicate lists of transcription factors and cofactors found in [Stampfel *et al.*](https://pubmed.ncbi.nlm.nih.gov/26550828/) and [Hens *et al.*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3929264/)

## Dependencies
Nearly all code is written in Python and run with version 3.10.5. The remaining code is written in Bash shell scripts. The following Python libraries were used.

|Library|Version|
|---|---|
|matplotlib|3.5.1|
|NumPy|1.22.3|
|pandas|1.4.2|
|SciPy|1.11.3|
|scikit-bio|0.5.7|
|scikit-learn|1.1.3|

Scikit-Bio attempts to import some deprecated warnings from scipy.stats during import, so these lines were commented out to ensure compatibility.

Additionally, the following Python packages were used for calculating sequence features.

|Package|Version|Source|Use|
|---|---|---|---|
|ipc|1.0|http://isoelectric.org/|For calculating isoelectric points|
|localcider|0.1.19|PyPI|For calculating other special sequence features, *e.g.*, sequence charge decoration|
