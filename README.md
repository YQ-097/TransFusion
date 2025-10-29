# TransFusion: Transfer Learning-Driven Adaptive Fusion Network for Infrared and Visible Image
Yao Qian, Rongsheng An, Gang Liu*, Haojie Tang, Gang Xiao, Durga Prasad Bavirisetti

Published in: Infrared Physics & Technology

- [paper](https://www.sciencedirect.com/science/article/pii/S1350449525001999)

## Abstract
The image fusion algorithm based on deep learning possesses strong feature extraction capabilities and generalization. However, due to the uninterpretability of features in deep learning, the design of fusion strategies becomes quite challenging. To address this issue, we propose a two-stage training feature adaptive fusion network based on the VGG-19 network. We introduce a parallel cross-modal channel perception module to achieve more targeted feature fusion by capturing channel differences between different modal domains. At the same time, in order to enhance the preservation of salient features, we designed a dynamic multi-level spatial attention guidance module that utilizes the saliency information of deep features from the source image to guide the adaptive fusion of shallow features. Additionally, we propose a double inner-loop feature mutual information loss that enforces the correlation of modal information, promoting efficient convergence of the perception module and guidance module. This method not only preserves the unique features of each modal domain but also effectively integrates information across modal domains, improving the quality of image fusion. Finally, we also perform objective and subjective experiments on MSRS and TNO datasets, and analyze the method. Experiments show that the proposed method achieves superior performance in image fusion tasks, and its potential value in practical applications is verified. 

## Framework
<img width="1609" height="775" alt="image" src="https://github.com/user-attachments/assets/9123ba91-0bfe-4862-b904-d9e6e8d23907" />

## Recommended Environment

 - [x] python 3.7 
 - [x] pytorch 1.12.1 
 - [x] scipy 1.2.1   
 - [x] numpy 1.11.3

## To Train
The training dataset is temporarily not publicly available. If needed, please contact the author for access.

    python train_stage1.py
    python train_stage2.py
## To Test
First, parameterize the structure of the trained model, and then run the testing program.

    test_image_stage1.py
    test_image_stage2.py
## Citation

```
@article{QIAN2025105906,
title = {TransFusion: Transfer learning-driven adaptive fusion network for infrared and visible image},
journal = {Infrared Physics & Technology},
volume = {150},
pages = {105906},
year = {2025},
issn = {1350-4495},
doi = {https://doi.org/10.1016/j.infrared.2025.105906},
url = {https://www.sciencedirect.com/science/article/pii/S1350449525001999},
author = {Yao Qian and Rongsheng An and Gang Liu and Haojie Tang and Gang Xiao and Durga Prasad Bavirisetti},
keywords = {Image fusion, Transfer learning, VGG-19, Feature fusion, Feature perception},
abstract = {The image fusion algorithm based on deep learning possesses strong feature extraction capabilities and generalization. However, due to the uninterpretability of features in deep learning, the design of fusion strategies becomes quite challenging. To address this issue, we propose a two-stage training feature adaptive fusion network based on the VGG-19 network. We introduce a parallel cross-modal channel perception module to achieve more targeted feature fusion by capturing channel differences between different modal domains. At the same time, in order to enhance the preservation of salient features, we designed a dynamic multi-level spatial attention guidance module that utilizes the saliency information of deep features from the source image to guide the adaptive fusion of shallow features. Additionally, we propose a double inner-loop feature mutual information loss that enforces the correlation of modal information, promoting efficient convergence of the perception module and guidance module. This method not only preserves the unique features of each modal domain but also effectively integrates information across modal domains, improving the quality of image fusion. Finally, we also perform objective and subjective experiments on MSRS and TNO datasets, and analyze the method. Experiments show that the proposed method achieves superior performance in image fusion tasks, and its potential value in practical applications is verified. The source code will be publicly available at https://github.com/YQ-097/TransFusion.}
}
```
