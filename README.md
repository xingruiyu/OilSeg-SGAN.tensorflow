# Adversarial f-Divergence Learning

Tensorflow implementation of TGRS paper [Oil Spill Segmentation via Adversarial f-Divergence Learning](https://ieeexplore.ieee.org/document/8301576/).  

We fetch lots of code from https://github.com/affinelayer/pix2pix-tensorflow. Thanks for their excellent project.

## Requirement
  CUDA 8.0  
  tensorflow 1.0.0  

## Usage
  run with  GAN  
  ```sh oilseg_gan.sh```  

  run with the proposed Adversarial f-Divergence Learning  
  ```sh oilseg_sgan.sh``` 

## If you find our method useful, please feel free to cite our paper.   

```bash
@ARTICLE{8301576, 
author={X. {Yu} and H. {Zhang} and C. {Luo} and H. {Qi} and P. {Ren}}, 
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
title={Oil Spill Segmentation via Adversarial$f$-Divergence Learning}, 
year={2018}, 
volume={56}, 
number={9}, 
pages={4973-4988}, 
keywords={Oils;Image segmentation;Synthetic aperture radar;Minimization;Training;Generators;Manuals;Adversarial learning;f-divergence minimization;oil spill segmentation;synthetic aperture radar (SAR) image processing}, 
doi={10.1109/TGRS.2018.2803038}, 
ISSN={0196-2892}, 
month={Sep.},}
```
