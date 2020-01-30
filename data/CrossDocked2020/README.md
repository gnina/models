# This directory contains all of the raw data for the CrossDocked2020 set

 * CrossDocked2020_types.tar.gz - Compressed directory containing all of the types files used to train models
 * TODO: add other tarballs once generated
 * The datafiles are currently being built and should be available by Feb 3rd -- Thank you for your patience!

## Downloading the tarballs
http://bits.csb.pitt.edu/files/crossdock2020/

You'll need to download CrossDocked2020_types.tar.gz and

## Extracting the tarballs
```
tar -xzvf CrossDocked2020_types.tar.gz
```

## Data structre
The raw data files are organized by Pockets. Each pocket is a directory which contains the following files:
```
TODO: add the files
```
We provide all of the data that was generated. Note: this is a superset to what was used in the paper. We provided all generated poses in order to help support the community in resampling this dataset. The types files specify the specific poses that we used in the paper (selected such that each pocket:ligand pose is distinct from the rest of them),

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
