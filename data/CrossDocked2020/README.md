# This directory contains all of the raw data for the CrossDocked2020 set

 * CrossDocked2020_types.tar.gz - Compressed directory containing all of the types files used to train models
 * CrossDocked2020.tgz          - Compressed directory containing all of the raw data.
 * crossdock2020_lig.molcache2  - molcache formatted version of the Ligand training data. Compatible with https://github.com/gnina/libmolgrid
 * crossdock2020_rec.molcache2  - molcache formatted version of the Receptor training data. Compatible with https://github.com/gnina/libmolgrid

## Downloading the tarballs
http://bits.csb.pitt.edu/files/crossdock2020/

You'll need to download CrossDocked2020_types.tar.gz and CrossDocked2020.tgz

## Extracting the tarballs
```
tar -xzvf CrossDocked2020_types.tar.gz
tar -xzvf CrossDocked2020.tgz
```

## Data structre
The raw data files are organized by Pockets. Each pocket is a directory which contains the following files:
```
<PDBid>_<chain>_rec.pdb                                                  -- Receptor file downloaded from the PDB
<PDBid>_<ligname>_uff2.sdf                                               -- UFF minimized version of the crystal pose from the PDB
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_tt_min.sdf             -- Autodock Vina minimized version of the ligand pose in the given receptor.
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_tt_docked.sdf          -- Autodock Vina docked poses of the ligand into the given receptor.
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_it1_tt_docked.sdf      -- First iteration CNN optimized poses of the original Vina docked poses
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_it1_it2_tt_docked.sdf  -- Second iteration CNN optimized poses on the first iteration CNN optimized poses
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_it2_tt_docked.sdf      -- Second iteration CNN optimized poses from the original Vina docked poses
```
We provide all of the data that was generated. Note: this is a superset to what was used in the paper. We provided all generated poses in order to help support the community in resampling this dataset. The types files specify the specific poses that we used in the paper (selected such that each pocket:ligand pose is distinct from the rest of them). 

## Types file format
There are 4 sets of the CrossDocked2020 data present here: ReDocked2020, CrossDocked2020, CrossDocked2020 It0, and CrossDocked2020 cross-docked only (cdonly) poses. CCV stands for the clustered-cross validation splits. 
```
CCV for ReDocked2020                                     -- types/it2_redocked_tt_*types
CCV for CrossDocked2020                                  -- types/it2_tt_0_*types
CCV for CrossDocked2020 without counter-example poses    -- types/it0_tt_0_*types
CCV for only the cross-docked poses                      -- types/cdonly_it2_tt*types
Train: all CrossDocked2020, Test: all CrossDocked2020    -- types/it2_tt_completeset_*.types
CCV for CrossDocked2020 (compatible with DenseNet)       -- types/mod_it2_tt_*types
```
See https://github.com/gnina/scripts for how to train a model using a given types file. It is highly recommended that you utilize the provided molcache2 files when training. This will help both in speed of loading the dataset and in efficient memory usage if training multiple models with the same machine.