# Facial Expression Recognition (FER) project by Andrew Farley for ELEC872.

This is a PyTorch project, the corresponding Conda environment can be found in ```env.yml```.

## Training
Once you have the Conda environment setup, you must edit the config file. The main config file used here is config_fer_semi.json. In this file you must specify the training, SSL, and validation datasets. Then, the model parameters can be changed (i.e. initial learning rate, maximum epochs, checkpoint path, etc.). To train a model run ```python train.py config_fer_semi.json```.

## Validation
Validation is similar to training. Once your config file is setup run ```python validate.py config_fer_semi.json -1 -1``` if you want to only validate the checkpoint specified in the config or ```python validate.py config_fer_semi.json a b``` to run validation from epoch a to epoch b (inclusive).
