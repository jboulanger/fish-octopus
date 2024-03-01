# RNA expression levels in the brains of paralarval Octopus vulgaris

Image analysis pipeline to quantify and spatially localise RNA expression levels in the brains of paralarval Octopus vulgaris, using in situ hybridisation chain reaction (HCR).

Nuclei are segmented using cellpose with the 'nuclei' pretrained model and the fish signal is quantified using a difference of Gaussian filter.

Workflow:
1. Run Select_ROI.ipynb to annotate the files
2. Process the ims files and the regions
3. Visualize the results and compute statistics.

## Files organization
- source folder
    - img1.ims
    - img2.ims
    - img3.ims

- destination
    - filelist.csv (after )
    - img1-regions.json
    - 
    
## Installation
```bash
git clone
```

```bash
# install micromamba
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
# reload the shell
${SHELL}
# create an environment
micromamba -qy create -f environment.yml
# activate the environment 
micromamba activate acourtney-fish-octopus

```


