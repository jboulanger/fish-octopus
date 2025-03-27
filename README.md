# RNA expression levels in the brains of paralarval Octopus vulgaris

[![DOI](https://zenodo.org/badge/946883361.svg)](https://doi.org/10.5281/zenodo.15008934)

Image analysis pipeline to quantify and spatially localise RNA expression levels in the brains of paralarval Octopus vulgaris, using in situ hybridisation chain reaction (HCR).

Nuclei are segmented using cellpose with the 'nuclei' pretrained model and the fish signal is quantified using a difference of Gaussian filter.

## Workflow on a workstation:
1. 1_Scan_folder.ipynb to create a list of files to work with
2. 2_Select_ROI.ipynb to define regions of interest
3. 3_Process.ipynb to process all files
3. 4_Visualization.ipynb to visualize the results and compute statistics.

## Remote processing:
### Configuration
- In a terminal connect to the login node, eg:
```bash
ssh hex
```
- get the code on the system, eg:
```bash
mkdir code
cd code
git clone https://github.com/jboulanger/fish-octopus.git
```
### Processing files
- connect to hex via visual code to edit the file script.sh
- run
```bash
sbatch --array 10 script.sh
```
replacing 10 by the number of lines in the filelist.csv.


## Files organization
- source folder
    - img1.ims
    - img2.ims
    - img3.ims

- destination
    - filelist.csv (after step 1)
    - img1-regions.json (after step 2)
    - img1-labels.tif (after step 3)
    - img1-measurements.csv (after step 3)
    - img1-stats.csv (after step 4)
    - ... (for files img2.ims, img3.ims)
    
## Installation

To create an environment from scratch:
```bash
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# reload the shell
${SHELL}
# Clone the repository
git clone https://github.com/jboulanger/fish-octopus
cd fish-octopus
# create an environment
micromamba -qy create -f environment.yml
# activate the environment 
micromamba activate acourtney-fish-octopus
```


