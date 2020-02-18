# Temporal Guassian Mixture Model trained on Vidor

## Overview

This repo is the build for the 2nd task of VidVRD: Video Relation Prediction

The 2st stage project: Video Action Detection

[The Grand Challenge MM2019](http://lms.comp.nus.edu.sg/research/dataset.html) 

## Download

[Vidor](http://lms.comp.nus.edu.sg/research/dataset.html)

### Extract the Features
To use this repository, you first need to extract the Vidor video segment features following the other repo [Vidor-i3d](https://github.com/Robbie-Luo/vidor-i3d)

## Train TGM
[train_tgm.py](train_tgm.py) 
contains the updated codes for training the tgm now compatiable with python 3.

E.g.
```bash
python train_tgm.py 
```
[vidor_i3d_per_video.py](vidor_i3d_per_video.py) contains the codes for loading extracted features from Vidor Dataset




