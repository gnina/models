This directory includes the models and atom maps used in our paper [preprint](https://doi.org/10.26434/chemrxiv.11833323.v1)

* default2017.model - the old default model for gnina
* default2017_norec.model - the old default model for gnina, for training ligand-only models
* default2018.model - the current default model
* default2018_norec.model - the current default model, for training ligand-only models
* hires_pose.model - a higher resolution (slower) model that performs particularly well at pose selection
* hires_pose_norec.model - a higher resolution (slower) model that performs particularly well at pose selection, for ligand-only models
* hires_affinity.model - a higher resolution (slower) model taht performs particularly well at affinity prediction
* hires_affinity_norec.model - a higher resolution (slower) model taht performs particularly well at affinity prediction, for ligand-only models
* dense.model - a densely connected CNN architecture from [Gao Huang et al](https://arxiv.org/abs/1608.06993)

## Usage
These model files are blank slates for training your own models.

In order to properly train a new model the following needs to change within the model file:
```
LIGCACHE_FILE -- needs to be changed to a *_lig.molcache2 file or delete the line.
RECCACHE_FILE -- needs to be changed to a *_rec.molcache2 file or delete the line.
```

Then you can proceed with train.py in gnina/scripts.
