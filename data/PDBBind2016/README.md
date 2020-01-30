# PDBBind version 2016 for training CNN models

 * PDBBind2016.tar.gz - Compressed folders containing the raw data in various formats
 * caches.tar.gz - Compressed files containing the Libmolgrid compatible caches for faster data loading.
 * \*.types - Files containing a given train/test split of the data.

## Downloading the missing tarball
http://bits.csb.pitt.edu/files/crossdock2020/

You need to download PDDBbind2016.tar.gz and PDBBind2016_caches.tar.gz

## Extracting the tarballs
```
tar -xzvf PDBBind2016.tar.gz
tar -xzvf caches.tar.gz
```

## Data structure
The tarball is organized by PDBid. Each receptor is a directory which contains the following files:
```
<PDBid>_ligand.<sdf/mol2>       -- Ligand file provided by the PDBbind
<PDBid>_protein.pdb             -- Receptor structure file provided by PDBbind
<PDBid>_pocket.pdb              -- Receptor binding pocket provided by PDBbind
<PDBid>_nowat.pdb               -- Receptor structure with all HETATOMS removed
<PDBid>_<ligand id>.sdf_allligs -- All ligands in the PDB with the given ligand ID, sdf format
<PDBid>_pdb.sdf                 -- Ligand structure for the given receptor, extracted from *.sdf_allligs. This is the crystal pose
<PDBid>uff.fail.sdf             -- File that only exists if the UFF via RDkit rejected the molecule
<PDBid>_uff.sdf                 -- UFF minimized pose of *_pdb.sdf
<PDBid>_min.sdf                 -- smina minimized pose of *_uff.sdf
<PDBid>_docked.sdf              -- smina docked pose of *_uff.sdf into its cognate receptor.
<PDBid>_crystal_0.gninatypes    -- gninatypes formatted version of the crystal pose. Compatible with libmolgrid
<PDBid>_docked_*.gninatypes     -- gninatypes formatted version of the *_docked.sdf. Compatible with libmolgrid
<PDBid>_min_0.gninatypes        -- gninatypes formatted version of the *_min.sdf. Compatible with libmolgrid
<PDBid>_ligand.smi              -- SMILES of the ligand
<PDBid>_conf.sdf                -- RDkit generated conformer from the ligand SMILES.
```

## Types file format
There are 3 sets of data present here: the PDBBind General, Refined, and Core sets.
Additionally, there are multiple different kinds of splits of the data present here.
The types files will segregate the data accordingly.

Each of the types files can be found in their own folder: Refined_types, General_types

### Refined_types
```
Clustered Cross Validated Splits                        -- ccv_ref_uff*
Train: Refined-Core Test:Core  with Docked poses        -- ref_uff_*
Train: Refined-Core Test:Core  with Crystal poses       -- ref2_crystal_*
Train: Refined+Core Test:Refined+Core with Docked poses -- fixed_ref_uff_completeset_*
```

### General_types
```
Clustered Cross Validated Splits                        -- ccv_gen_uff*
Clustered Cross Validated Splits    Ligand Only         -- ccv_gen_norec_uff*
Train: Refined-Core Test:Core  with Docked poses        -- gen_uff_*
Train: Refined+Core Test:Refined+Core with Docked poses -- fixed_ref_uff_completeset_*
```
## Caches
```
gen2_crystal_*.molcache2   -- Cache containing all of the data's crystal poses
gen2_docked_uff*.molcache2 -- Cache containing all of the data's docked poses
```
