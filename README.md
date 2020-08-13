# Condensing Two-stage Detection with Automatic Object Key Part Discovery

This project implements [paper](https://arxiv.org/abs/2006.05597) which significantly reduces the amount of parameters required to achieve high accuracy for detection heads. We are still developing this idea. Please wait for our future works. 

This project is built upon the [mmdetection](https://github.com/open-mmlab/mmdetection) v2.0.0, mmcv==v0.6.2. Config files for training can be found in './configs/condense2stagedet/'.

Note that we originally use the previous version of mmdet and recently it was upgraded to >= v2.0.0. This causes a little difference in training results. For example, our current implementation does not work well when using L1 Loss rather than using Smooth L1 Loss for regression in the latest version of mmdet. 
