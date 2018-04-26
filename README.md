# ACT_test_theresult
BY Haoliang Tan XJTU Computer science 

## Introduction
The ACtion Tubelet detector (ACT-detector) is a framework for action localization. 
It takes as input sequences of frames and outputs tubelets, i.e., sequences of bounding boxes with associated scores.

For more details, please refer to our [ICCV 2017 paper](https://hal.inria.fr/hal-01519812/document) and our [website](http://thoth.inrialpes.fr/src/ACTdetector/). 

## Citing ACT-detector

If you find ACT-detector useful in your research, please cite: 

    @inproceedings{kalogeiton17iccv,
      TITLE = {Action Tubelet Detector for Spatio-Temporal Action Localization},
      AUTHOR = {Kalogeiton, Vicky and Weinzaepfel, Philippe and Ferrari, Vittorio and Schmid, Cordelia},
      YEAR = {2017},
      BOOKTITLE = {ICCV},
    }

## Contents
1. add_tube.py

## Datasets
How to use this code 
1.you need to extract the tubes and tubelests at first!   Yan can find here https://github.com/vkalogeiton/caffe/tree/act-detector
  but note that I only implement this project in Dataset {UCFSPORTS}
 
2.After you do the first step you should get the {pkl} (such as 000001.pkl)files for each frames in test videos and {pkl}files
   (such as 001_tubes.pkl) for each test videos. 
3.If you do not want to extract pkl files! (Because the step is very fussy the inefficiency), you can skip the first step dircetlly!
4. JUst look the add_tube.py file is OK ! 

## Enjoy it ! Study together ! 
