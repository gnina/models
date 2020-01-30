This directory includes the models and atom maps used in our paper.

* default2017.model - the old default model for gnina
* default2018.model - the current default model
* hires_pose.model - a higher resolution (slower) model that performs particularly well at pose selection
* hires_affinity.model - a higher resolution (slower) model taht performs particularly well at affinity prediction
* dense.model - a densely connected CNN architecture from [Gao Huang et al](https://arxiv.org/abs/1608.06993)

## Usage
These model files are blank slates for training your own models.

In order to properly train a new model the following needs to change within the model file:
```
LIGCACHE_FILE -- needs to be changed to a *_lig.molcache2 file or delete the line.
RECCACHE_FILE -- needs to be changed to a *_rec.molcache2 file or delete the line.
```

Then you can proceed with train.py in gnina/scripts.