# Repository for Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution

![aaaa](https://user-images.githubusercontent.com/59813678/230926803-948c2bfa-cb58-4c48-8e55-621dad4ed2a2.png)



**Abstract**: *This paper proposes a new pipeline to perform early action detection from skeleton-based untrimmed videos. Our pipeline includes two new technical components. The first is a new Dynamic Dilated Convolutional Network (DDCN), which supports dynamic temporal sampling and makes feature learning more robust against temporal scale variance in action sequences. 
The second is a new semantic referencing module, which uses identified objects in the scene and their co-existence relationship with actions to adjust the probabilities of inferred actions. Such semantic guidance can help distinguish many ambiguous actions, which is a core challenge in the early detection of incomplete actions. Our pipeline achieves state-of-the-art performance in early action detection in two widely used skeleton-based untrimmed video benchmarks.*


Please look at our Pattern Recognition paper:
[Link to our paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320323002960)

**Installing dependencies:**

Prerequisites:   
- Pytorch > 1.6 
- TensorBoard  
- Scikit-learn

(Optional):
- Detectron2  
- Spotlight  
- OpenCV

Use the following codes to install the main dependencies:
- ```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch```
- ```pip install tensorboard==2.4.0```
- ```pip install scikit-learn==1.0.2```

**Results on the OAD dataset:**

**Preparing data:**

We provide you the download links for the data and the pre-trained models. 
For the [OAD](https://www.icst.pku.edu.cn/struct/Projects/OAD.html) dataset download the data from [here](https://drive.google.com/file/d/1gVPZqDGZcQPLoxkRabi6b4NN09tIpszL/view?usp=sharing) and download the model from [here](https://drive.google.com/file/d/1tHmqnFbKi3UpEvAZTsSo6An969xTWp99/view?usp=sharing).   

Please extract the .zip files and copy the downloaded ```data``` and ```model``` folders to the the root folder of the source codes and fix the paths in ```run_script_OAD.py``` accordingly. 

**Evaluating the results:**

Please run ```Python run_script_OAD.py``` to output the action detection performances (F1 scores) for different Observation Ratios and modules (DDCN and SRM) on the *OAD* dataset.  

**Using SRM manually:**

We provide you the *SRM* outputs, *offline semantic reference attributes*, ```off_sem_ref_attr_OAD.npy```, and the *semantic reference scores*, ```sem_ref_scr_OAD.npy``` above.  

If you want to obtain the *SRM* outputs manually, please use the following codes in order in the ```SRM``` folder:  

1. ```convert_to_sem_ref_attr_OAD.py```: to convert the *OAD* data to *semantic reference attributes*.  
2. ```convert_to_sem_ref_scr_OAD.py```: to convert the *semantic reference attributes* to *semantic reference scores*, using *recommendation systems*.  

You need the [Detectron2](https://github.com/facebookresearch/detectron2), [Spotlight](https://github.com/maciejkula/spotlight), and [OpenCV](https://pypi.org/project/opencv-python/) libraries for the above manual conversion. 

**Results on the PKU-MMD dataset:**   

For the [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html) dataset download the data from [here](https://drive.google.com/file/d/1WxL5emkbwOUr_5ceQvq98dTC71AQVbRt/view?usp=sharing) and download the model from [here](https://drive.google.com/file/d/1-pOiHugpedRI3e9FDXBWfhlRszVTNgbi/view?usp=sharing).  
The data is from the *Cross-Subject Evaluation* set.  

Please extract the .zip files and copy the downloaded ```data``` and ```model``` folders to the the root folder of the source codes and fix the paths in ```run_script_PKU.py``` accordingly.    

Please run ```Python run_script_PKU.py``` to output the action detection performances (F1 scores) for different Observation Ratios on the *PKU* dataset.  

**Reference**

If you find this repository useful please cite us:

```
@article{korban2023DDCN,
  title={Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution},
  author={Korban, Matthew and Li, Xin},
  journal={Pattern Recognition},
  pages={109595},
  year={2023},
  publisher={Elsevier}
  }
  ```
  
  ```
  Korban, Matthew, and Xin Li. "Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution." Pattern Recognition (2023): 109595.
  ```


