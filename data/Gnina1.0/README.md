# This directory contains all of the raw data used for the evaluations in the [GNINA 1.0 manuscript](https://doi.org/10.1186/s13321-021-00522-2)
 - `crossdocked_all_data.tar.gz` - All of the proteins and ligands in the [Wierbowski _et al._  dataset](https://doi.org/10.1002/pro.3784) (note that this is different from the [CrossDocked2020 dataset](https://github.com/gnina/models/blob/master/data/CrossDocked2020/))
 - `crossdocked_ds_data.tar.gz` - The downsampled version of the [Wierbowski _et al._  dataset](https://doi.org/10.1002/pro.3784) used for the cross-docking evaluations in the paper
 - `redocking_all_data_tar.gz` - Protein and ligands used for the redocking evalutations in the paper
 - `redocking_docked_structures.tar.gz` - Docked structures from the evaluations carried out on the redocking dataset
 - `crossdocking_docked_structures.tar.gz` - Docked structures from the evaluations carried out on the cross-docking dataset
 - `rd_input_pairs.txt` - Space delimited file containing receptor, ligand, autobox_ligand, and output prefix for redocking data
 - `rd_wp_input_pairs.txt` - Space delimited file containing receptor, ligand, autobox_ligand(whole protein), and output prefix for redocking data 
 - `ds_cd_input_pairs.txt`- Space delimited file containing receptor, ligand, autobox_ligand, and output prefix for cross-docking data
 - `ds_cd_wp_input_pairs.txt` - Space delimited file containing receptor, ligand, autobox_ligand(whole protein), and output prefix for cross-docking data 
 - `rd_not_training.txt` - List of redocking PDB IDs utilized in the generalization evaluation of the paper
 - `cd_not_training.txt` - List of cross-docking PDB IDs utilized in the generalization evaluation of the paper
 
## Downloading the tarballs

https://bits.csb.pitt.edu/files/gnina1.0_paper/

If you would like to complete all of the docking in the Gnina1.0 paper, you'll need to download `crossdocked_ds_data.tar.gz` and `redocking_all_data.tar.gz`

If you would like all of the docked structures used in the evaluations in the Gnina1.0 paper, you'll need to download `redocking_docked_structures.tar.gz` and `crossdocking_docked_structures.tar.gz`

## Extracting the tarballs
 For running the docking and analysis in the paper:
 ```
 tar xzvf crossdocked_ds_data.tar.gz 
 tar xzvf redocking_all_data.tar.gz
 ```

 For running the analysis in the paper:
 ```
 tar xzvf crossdocking_docked_structures.tar.gz 
 tar xzvf redocking_docked_structures.tar.gz
 ```

## Data Structure of *_data.tar.gz

### Redocking
The files are organized by the PDB ID of the protein-ligand complex. Each PDB ID is a directory which contains the following files:
```bash
<PDB_ID>_PRO.pdb.gz       -- Receptor file downloaded from the PDB with all other atoms removed
<PDB_ID>_LIG.sdf.gz        -- Ligand file downloaded from the PDB
```

### Cross-docking
The files are organized by pockets. Each pocket a directory which contains the following files:
```bash
<PDB_ID>_PRO.pdb               -- Receptor file downloaded from the PDB
<PDB_ID>_LIG_aligned.sdf        -- Ligand file downloaded from the PDB
```
## Data Structure of *_docked_structures.tar.gz

### Redocking
The files are organized by the PDB ID of the protein-ligand complex. Each PDB ID is a directory which contains the following files:
```bash
<PDB_ID>_<autoblig|fullprot>_<cnn>_<cnn_scoring>_<other_options>.sdf.gz
```
`autoblig` indicates docking where the `autobox_ligand` is set as the actual binding pocket and `fullprot` indicates whole protein docking.  
`<cnn>` indicates the CNN model(s) utilized for the given `<cnn_scoring>` option.  
`<cnn_scoring>` will either be `refinement` or `rescore`  
`<other_options>` will either be `defaults` indicating that all default options of GNINA were used or one of the following options:
 - `cnn_empirical_weight`
 - `exhaustiveness`
 - `min_rmsd_filter`
 - `num_modes`
 - `cnn_rotation`
 - `autobox_add`

 which is followed immediately by a number indicating the value the parameter was set to (all other parameters kept at defaults)

### Cross-docking
The files are organized by pockets. Each pocket a directory which contains the following files:
```bash
<PDB_ID_PRO>_PRO_<PDB_ID_LIG>_LIG_aligned_<wp_>v2_<autoblig|fullprot>_<cnn>_<cnn_scoring>_<other_options>.sdf.gz
```
If `wp_` precedes `v2` then the docked structures were generated via whole protein docking, otherwise the docking utilized the native ligand for the `autobox_ligand`  
`<cnn>` indicates the CNN model(s) utilized for the given `<cnn_scoring>` option.  
`<cnn_scoring>` will either be `refinement` or `rescore`  
`<other_options>` will either be `defaults` indicating that all default options of GNINA were used or one of the following options:
 - `cnn_empirical_weight`
 - `exhaustiveness`
 - `min_rmsd_filter`
 - `num_modes`
 - `cnn_rotation`
 - `autobox_add`

 which is followed immediately by a number indicating the value the parameter was set to (all other parameters kept at defaults)

## txt file format
Each of the txt files provided have the following structure:
```bash
<location of protein file> <location of docking ligand> <location of autobox_ligand specifier> <prefix of output file name>
```

These txt files are designed to be utilized with [make_gnina_cmds.py](https://github.com/dkoes/GNINA-1.0/blob/main/analysis_scripts/make_gnina_cmds.py) to make a file with GNINA commands for running all of the experiments in the paper.

## Analysis
Information about the analysis pipeline as well as the figure generation can be found in the [GNINA1.0 Repo](https://github.com/dkoes/GNINA-1.0/). All of the python scripts used for the analysis in our paper can be found in [analysis_scripts](https://github.com/dkoes/GNINA-1.0/blob/main/analysis_scripts/) in that repo as well as comprehensive instructions about how to run the analysis.
