# Repository for Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution


**Installing dependencies:**

Prerequisites:   
- Pytorch > 1.6 
- TensorBoard  
- Scikit-learn

(Optional):
- Detectron2  
- Spotlight  
- OpenCV

Use the following codes to install the dependencies:
- ```conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch```
- ```pip install tensorboard==2.4.0```
- ```pip install scikit-learn==1.0.2```

**Preparing data:**

We provide you the download links for the data and the pre-trained models. 
For the [OAD](https://www.icst.pku.edu.cn/struct/Projects/OAD.html) dataset download the data from [here](https://drive.google.com/file/d/1gVPZqDGZcQPLoxkRabi6b4NN09tIpszL/view?usp=sharing) and download the model from [here](https://drive.google.com/file/d/1tHmqnFbKi3UpEvAZTsSo6An969xTWp99/view?usp=sharing).   

Please extract the .zip files and copy the downloaded ```data``` and ```model``` folders to the the root folder of the source codes and fix the paths in ```run_script_OAD.py``` accordingly. 

**Evaluating the results:**

Please run ```Python run_script_OAD.py``` to output the action detection performances (F1 scores) for different Observation Ratios and modules (DDCN and SRM) on the *OAD* dataset.  

**Using SRM manually:**

We provide you the *SRM* outputs, *offline semantic reference attributes*, ```off_sem_ref_attr_OAD.npy```, and the *semantic reference scores*, ```sem_ref_scr_OAD.npy``` above.  

If you want to obtain the *SRM* outputs manually, please use the following codes in order in the ```SRM``` folder:  

1. ```convert_to_sem_ref_attr_OAD.py```: to convert the OAD data to *semantic reference attributes*.  
2. ```convert_to_sem_ref_scr_OAD.py```: to convert the *semantic reference attributes* to *semantic reference scores*, using *recommendation systems*.  

You need the [Detectron2](https://github.com/facebookresearch/detectron2), [Spotlight](https://github.com/maciejkula/spotlight), and [OpenCV](https://pypi.org/project/opencv-python/) libraries for the above manual conversion. 

The source codes and data for PKU-MMD dataset will be released soon. 
