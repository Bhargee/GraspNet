# GraspNet

Code, data, and models for graspnet project v2. Please do not touch without
first asking Bhargava or Joe.

# Layout
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

# Networks
Using vgg{16,19} and resnet networks, both recent ilsrvc winners

# Training notes
Read [this paper](http://cs231n.stanford.edu/reports2016/405_Report.pdf) on fine tuning  
on vgg and resnet networks and [the original vgg paper](https://arxiv.org/pdf/1409.1556.pdf)

Will augment training sets with random flips
**potential problem** - training images are nonsquare, net input is square, not optimal combo
