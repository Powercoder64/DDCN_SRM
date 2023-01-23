# Rrepository for Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution


**Prerequisites:**

- Pytorch > 1.4  
- TensorBoard  
- NumPy

(Optional):
- Detectron2  
- Spotlight  
- OpenCv

Please run ```Python run_script_OAD.py``` to output the action detection performances (F1 scores) for different Observation Ratios and modules (DDCN and SRM) on the [OAD](https://www.icst.pku.edu.cn/struct/Projects/OAD.html) dataset.  


We provide you the download links for the data and the pre-trained models. 
For the OAD dataset download the data from [here](https://drive.google.com/file/d/1gVPZqDGZcQPLoxkRabi6b4NN09tIpszL/view?usp=sharing) and download the model from [here](https://drive.google.com/file/d/1tHmqnFbKi3UpEvAZTsSo6An969xTWp99/view?usp=sharing).   


Please extract the .zip files and copy the downloaded ```data``` and ```model``` folders to the the root folder of the source codes and fix the paths in ```run_script_OAD.py``` accordingly. 

We provide you the *SRM* outputs, *offline semantic reference attributes*, ```off_sem_ref_attr_OAD.npy```, and the *semantic reference scores*, ```sem_ref_scr_OAD.npy```.  

If you want to obtain the above *SRM* outputs manually, please use the following codes in the ```SRM``` folder:  

```convert_to_sem_ref_attr_OAD.py```: to convert the OAD data to *semantic reference attributes*.  
```convert_to_sem_ref_scr_OAD.py```: to convert the *semantic reference attributes* to *semantic reference scores*, using *recommendation systems*.  

You need the [Detectron2](https://github.com/facebookresearch/detectron2) and [Spotlight](https://github.com/maciejkula/spotlight) libraries for the above manual conversion. 

The source codes and data for PKU-MMD dataset will be released soon. 
