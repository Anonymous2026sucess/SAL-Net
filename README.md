# SAL-Net

## 0. Abstract

 The Kolmogorov-Arnold Network (KAN) has attracted wide-spread attention in the field of medical image segmentation due to its powerful nonlinear modeling capabilities. However, existing KAN-based methods mostly use it as a backbone network, which can easily lead to information loss during downsampling and global context modeling. Furthermore, the skip connections in these methods often use simple addition operations to fuse encoder outputs, which makes it difficult to fully integrate contextual information. To address this issue, we propose SAL-Net: We introduce a state-space interactive attention mechanism into the KAN backbone network to achieve efficient global context modeling. We also design a large-kernel selective attention module to adaptively fuse spatial features at different scales, thereby learning the optimal receptive field and improving the recognition of diverse objects. Comprehensive experimental results on the BUSI, GlaS, and ISIC2017 datasets demonstrate that SAL-Net consistently outperforms existing methods, achieving superior segmentation accuracy and robustness. The code is publicly available at [https://github.com/szz25/SAL-Net.


## 1. Overview

<div align="center">
<img src="Figs/SAL-Net.png" />
</div>



## 2. Main Environments

The environment installation process can be carried out as follows:

```
conda create -n SAL-Net python=3.8
conda activate SAL-Net
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  
pip install mamba_ssm==1.0.1
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```



## 3. Datasets

You can refer to [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for processing datasets, but for the division of the PH2 dataset, please run the Prepare_PH2.py we provide to divide the training set, validation set, and test set. Then organize the .npy file into the following format:

'./datasets/'

- ISIC2017
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- ISIC2018
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy
- PH2
  - data_train.npy
  - data_val.npy
  - data_test.npy
  - mask_train.npy
  - mask_val.npy
  - mask_test.npy



## 4. Train the SAL-Net

```
python train.py
```



## 5. Test the SAL-Net 

First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.

```
python test.py
```



## 6. Comparison With State of the Arts

The performance of the proposed method is compared with the state-of-the-art models on the ISIC2017, ISIC2018, and $\text{PH}^2$ datasets, with the top two results highlighted in red and blue, respectively.

<div align="center">
<img src="Figs/Table1.png" />
</div>



## 7. Acknowledgement

Thanks to [Vim](https://github.com/hustvl/Vim), [VM-UNet](https://github.com/JCruan519/VM-UNet) and [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet) for their outstanding works.
