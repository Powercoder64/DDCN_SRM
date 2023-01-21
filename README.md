# Rrepository for Semantics-enhanced Early Action Detection using Dynamic Dilated Convolution


**Prerequisites:**

- Pytorch > 1.4  
- TensorBoard  
- NumPy

(Optional):
- Detectron2  
- Spotlight  

Please run the file *run_script_OAD.py* to output the action detection performances (F1 scores) for different Observation Ratios and modules (DDCN and SRM) on the OAD dataset.  


We provide you the download links for data and the pre-trained model for the OAD dataset here:  
[OAD_data](https://drive.google.com/file/d/1gVPZqDGZcQPLoxkRabi6b4NN09tIpszL/view?usp=sharing)  
[DDCN_model_OAD](https://drive.google.com/file/d/1tHmqnFbKi3UpEvAZTsSo6An969xTWp99/view?usp=sharing).  

Please extract the zip files and copy the downloaded *data* and *model* folders the the root folder of the source codes and fix the paths in *run_script_OAD.py* accordingly. 

We provide you the offline semantic reference attributes, *off_sem_ref_attr_OAD.npy*, and semantic reference scores, *sem_ref_scr_OAD.npy*.  

If you want to obtain the above SRM outputs manually, please use the following files in the *utils* folder:  

*convert_to_sem_ref_attr.py*: to onvert the OAD data to semantic reference attributes.  
*convert_to_sem_ref_scr.py*: to convert the semantic reference attributes to semantic reference scores, using recommendations system: IMF and others.  
You need the *Detectron2* and *Spotlight* libraries for the above. 

The source codes and data for PKU-MMD dataset will be released soon. 
