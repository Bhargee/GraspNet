# GraspNet

Code, data, and models for graspnet project v2. Please do not touch without
first asking Bhargava or Joe.

# Instructions
1. install caffe locally in this repository, under `caffe`
2. copy relevant data in the correct subdirectories under origdata
3. download pretrained models to correct subdirectories in models
4. read `preprocess_data.py`, make modifications as necessary, and run
5. run `create_dataset.sh`
6. run `make_dataset_mean.sh`

# Layout
After steps 1 and 2 above are done, the project structure should look like - 
```
root/
  caffe/ <- local install of caffe  
  models/ <- where model definitions and trained weight files go  
  tarballs/ <- zipped original images for 2 datasets, just in case  
  tensorflow/ <- local tensorflow install in virtualenv   
  origdata/  
    HandCam/  
      Images/  
      Anno_HandCam.json  
    ImageNet/ <- *curated imagenet*  
      Images/  
      Anno_ImageNet.json  
    DeepGrasping/  
      Images/
      Anno_DeepGrasping.json  
    AllImages/ <- symlink all files under Images above here    
    
   trainingdata/ <- preprocessed data for training nets, with train and test text files  
       train/  
       val/  
```
# Networks
Using vgg{16,19} and resnet networks, both recent ilsrvc winners

# Training notes
Read [this paper](http://cs231n.stanford.edu/reports2016/405_Report.pdf) on fine tuning  
on vgg and resnet networks and [the original vgg paper](https://arxiv.org/pdf/1409.1556.pdf)

Will augment training sets with random flips
**potential problem** - training images are nonsquare, net input is square, not optimal combo
