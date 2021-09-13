# CoReD: Generalizing Fake Media Detection with Continual Representation using Distillation (ACMMM'21 Oral Paper)
    
**(Accepted for oral presentation at [ACMMM '21](https://2021.acmmm.org/))**

Paper Link: ([arXiv](https://arxiv.org/abs/2107.02408)) (ACMMM version (soon))
<p align="center">
<img src="imgs/Overview.PNG" alt="CLRNet-pipeline" border="0" width="600">


## Overview
We propose _Continual Representation using Distillation_ (_CoReD_) method that employs the concept of **Continual Learning (CL)**, **Representation Learning (RL)**, and **Knowledge Distillation (KD)**.
 
### Comparison Baselines
- Transfer-Learning (_TL_) : The first method is Transfer learning, where we perform fine-tuning on the model to learning the new Task.  
- Distillaion Loss (_DL_) : The third method is a part of our ablation study, wherewe only use the distillation loss component from our CoReD loss function to perform incremental learning.  
- Transferable GAN-generated Images Detection Framewor (_TG_) : The second method is a KD-based GAN image detection framework using L2-SP and self-training.  
    
## Training & Evaluation
   
### - Requirements and Installation
We recommend the installation using the _requilrements.txt_ contained in this Github. 
    
python==3.8.0  
torchvision==0.9.1  
torch==1.8.1  
sklearn  
numpy  
opencv_python  
    
```console
pip install -r requirements.txt
```
    
### - Full Usages
```console
tbd
```

####
- **Note that** 

### - Training & Evaluation
To train and evaluate the model(s) in the paper, run this command:
- **Task1**
    ```TRAIN
   python 
    ```
   After train the model, you can evaluate the result.
    ```EVAL
    python 
    ```


## Result
- **AUC scores (%)** of various methods on compared datasets.
#### - Task1 (GAN datasets and FaceForensics++ datasets)
<div style="text-align:center">
    <img src="./imgs/task1_gan.png" width="350"/> <img src="./imgs/task1_ff.png" width="350"/>
</div>

#### - Task2 - 4
<div style="text-align:center">
    <img src="./imgs/task2.png" width="350"/>
</div>

<div style="text-align:center">
    <img src="./imgs/task3.png" width="350"/>
</div>

<div style="text-align:center">
    <img src="./imgs/task4.png" width="350"/>
</div>

## Citation
If you find our work useful for your research, please consider citing the following papers :)
```
@misc{kim2021cored,
    title={CoReD: Generalizing Fake Media Detection with Continual Representation using Distillation},
    author={Minha Kim and Shahroz Tariq and Simon S. Woo},
    year={2021},
    eprint={2107.02408},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
 
## - Contect
If you have any questions, please contact us at **kimminha/shahroz@g.skku.edu**
 
## References
###### [1] 


## - License
The code is released under the MIT license.
Copyright (c) 2021




