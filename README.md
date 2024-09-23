<div>
<img src="logo.png" align="left" style="margin: 10 10 10 10;" height="150px">
	

<strong>ABCL-Net</strong>: 

<b><u>A</u></b>ttention-<b><u>b</u></b>ased <b><u>C</u></b>ollaborative <b><u>L</u></b>earning Network for Joint Modality Generation and Tissue Segmentation of Early-developing Macaque Brain MR Images.
</div>
<br />


<br />
<br />

# Introduction
The ABCL-Net, designed for multi-task cooperation and cross-modality feature interchange, excels at performing both missing modality generation and tissue segmentation simultaneously. By inputting one T1w (T2w) image, users can obtain one generated T2w (T1w) image and one predicted label for macaque brain tissue segmentation.
![](https://github.com/XueyangWWW/ABCL-Net/blob/main/Figure1.png)

# System requirement
Since this is a *Linux* based container, please install the container on a Linux system. The supported systems include but not limited to `Ubuntu`, `Debian` and `CentOS`. 

The pipeline is developed based on deep convolutional neural network techniques. A GPU (≥10GB) is required to support the processing. 
