# This directory contains instruction to download the raw data for the CrossDocked2020 set

 * CrossDocked2020_v1.3_types.tar.gz    - Compressed directory containing all of the types files used to train models
 * CrossDocked2020_v1.3.tgz             - Compressed directory containing all of the raw data. This tarball contains 52126979 files
 * crossdock2020_1.3_lig.molcache2      - molcache formatted version of the Ligand training data. Compatible with https://github.com/gnina/libmolgrid
 * crossdock2020_1.3_rec.molcache2      - molcache formatted version of the Receptor training data. Compatible with https://github.com/gnina/libmolgrid
 * downsampled_CrossDocked2020_v1.3_types.tgz - Compressed directory containing the types files for the downsampled set to train models.
 * downsampled_CrossDocked2020_v1.3.tgz       - Compressed directory containing all of the raw data that is present in our downsampled set.

## Changelog
Version 1.3 of CrossDocked2020 has been released. We initiated a new pocket downloading schema where a ligand was only downloaded if it has a different cognate receptor (e.g. only 1 ligand file will be produced for PDB a1b2 even if a1b2 chain A and a1b2 chain B both have ligand bound). Additionally, we changed the parsing of pocketome to only download the 'non-redundant set' which helped removed some data entries that were not providing meaningful information to be used when training the model. Additionally, version 1.3 has addressed several receptor and ligand structures which were mis-aligned, and fixed other ligands which had the wrong bond typing on various aromatic bonds.

As approximately 60 percent of the dataset was re-generated, we recommend redownloading the types files from the server.

[//]: # (Version 1.2 of CrossDocked2020 was released. It addressed several receptor and ligand structures in version 1.1 that had their aromatic rings removed. Version 1.2 has addressed these issues. Since this issue was pervasive, we recommend redownloading the types files from the server.)

## Downloading the tarballs
http://bits.csb.pitt.edu/files/crossdock2020/

You'll need to download CrossDocked2020_v1.3_types.tgz and CrossDocked2020_v1.3.tgz

## Extracting the tarballs
WARNING -- CrossDocked2020_v1.3.tgz contains 52,126,979 files!! It will use a lot of space and take a long time!
```
wget http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3.tgz
wget http://bits.csb.pitt.edu/files/crossdock2020/CrossDocked2020_v1.3_types.tgz
mkdir CrossDocked2020
tar -xzvf CrossDocked2020_v1.3_types.tgz
tar -C CrossDocked2020 -xzf CrossDocked2020_v1.3.tgz
```

## Data structre
The raw data files are organized by Pockets. Each pocket is a directory which contains the following files:
```
<PDBid>_<chain>_rec.pdb                                                              -- Receptor file downloaded from the PDB
<PDBid>_<chain>_lig.pdb                                                              -- Ligand file downloaded from the PDB
<PDBid>_<ligname>_uff<2>.sdf                                                         -- If possible, a UFF minimized version of the crystal pose from the PDB
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_tt_min.sdf.gz                         -- Autodock Vina minimized version of the ligand pose in the given receptor.
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_tt_docked.sdf.gz                      -- Autodock Vina docked poses of the ligand into the given receptor.
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_it1_tt_docked.sdf.gz                  -- First iteration CNN optimized poses of the original Vina docked poses
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_<it1_it2 | it2_it1>_tt_docked.sdf.gz  -- Second iteration CNN optimized poses on the first iteration CNN optimized poses
<rec PDBid>_<chain>_rec_<lig PDBid>_<ligname>_lig_it2_tt_docked.sdf.gz                  -- Second iteration CNN optimized poses from the original Vina docked poses
<prefix>_<pose>.gninatypes                                                              -- Gninatypes format of the file corresponding to the Prefix.
```
We provide all of the data that was generated. Note: this is a superset to what was used in the paper. We provided all generated poses in order to help support the community in resampling this dataset. The types files specify the specific poses that we used in the paper (selected such that each pocket:ligand pose is distinct from the rest of them). 

## Types file naming convention
There are 4 sets of the CrossDocked2020 data present here: ReDocked2020, CrossDocked2020, CrossDocked2020 It0, and CrossDocked2020 cross-docked only (cdonly) poses. CCV stands for the clustered-cross validation splits. 
```
it2           -- CrossDocked2020 with 2 rounds of iteratively generated counterexamples in 3 fold CCV format
it0           -- CrossDocked2020 without counterexamples in 3 fold CCV format
mod_          -- prefix for modified it2 types files for the DenseNet (don't have the <RMSD to crystal> column
_completeset_ -- All of the CrossDocked2020 it2 poses in a single training/testing file
cdonly_       -- CrossDocked2020 with only the cross-docked poses present
_redocked_    -- CrossDocked2020 with only the re-docked poses present
```
See https://github.com/gnina/scripts for how to train a model using a given types file. It is highly recommended that you utilize the provided molcache2 files when training. This will help both in speed of loading the dataset and in efficient memory usage if training multiple models with the same machine.

## Types file format
Each of the types files utilized here have the following structure:
```
<label> <pK> <RMSD to crystal> <Receptor> <Ligand> # <Autodock Vina score>
```
Where the label is 1 if the RMSD to the crystal pose is <=2, and 0 otherwise. The pK is calculated by taking the negative log (base 10) of the given number in the PDBbind. We made no distinction between Kd/Ki/IC50. IF the ligand has an unknown affinity, then it is recorded as 0. We additionally labeled the binding affinities as negative if the pose is >2 (this makes it easier to identify the poor pose for our network's hinge loss). The receptor and ligand columns correspond to the filenames of the raw data file (and are utilized with our molcahce files). 

NOTE: the exception to this is the types files for the DenseNet, which has the RMSD column removed.

## Getting models running with LIBMOLGRID
If you are utilizing [libmolgrid](https://github.com/gnina/libmolgrid) to train models and wish to use this data, we have provided molcaches which contain all of the data and are much smaller.

This is especially handy, as the data can be used as is, without the need to download the entire CrossDocked2020 raw datafiles

```
wget http://bits.csb.pitt.edu/files/crossdock2020/crossdock2020_1.3_rec.molcache2
wget http://bits.csb.pitt.edu/files/crossdock2020/crossdock2020_1.3_lig.molcache2
```

These caches are usable with our types files (input files to Caffe which define the training data).

## Using the Downsampled set instead
Even when using just molcaches and types files to train models, there is still about 22Gb of data that needs to be loaded into memory. In order to provide a less intesive version of the dataset, we also provide a downsampled version of CrossDocked2020.

```
wget http://bits.csb.pitt.edu/files/crossdock2020/downsampled_crossdock2020_1.3_rec.molcache2
wget http://bits.csb.pitt.edu/files/crossdock2020/downsampled_crossdock2020_1.3_lig.molcache2
```

These caches are utilized by a different set of types files:

```
wget http://bits.csb.pitt.edu/files/crossdock2020/downsampled_CrossDocked2020_v1.3_types.tgz
tar -xzf downsampled_CrossDocked2020_v1.3_types.tgz
```

We have provided a downsampled version of the it2 clustered cross-validated sets mentioned above. This version only requres about 5Gb of data to be loaded into memory.

WARNING -- Each file was sampled independently, by taking 10 good (<2 RMSD) and 20 poor (>2 RMSD) poses for each Pocket:Ligand pair! This means that train0+test0 do NOT contain the same poses as train1+test1.

The downsampled sets provdided here were generated with the intelligent_downsample.py script available at https://github.com/dkoes/cnnaffinitypaper
