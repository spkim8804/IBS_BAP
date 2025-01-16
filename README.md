# IBS Behavior Analysis Program (IBS-BAP)
* IBS-BAP is a project aimed at creating a powerful and convenient behavior analysis program.
* The ultimate goal is to enable one-step behavior analysis, from classification to analysis, directly from images or videos.
* Currently, it supports pose estimation using YOLO11, as well as visualization and management of bounding box label data.
* More features will be updated soon!

## Features
Visualize image file (jpg, png) and video file (mp4) and make bounding boxes. In case of mp4 file, if it is not **i-frame video**, the program will offer you to convert your video (Generate new mp4 file with "_iframe" in the same directory)

Run Yolo11 for pose estimation (You can use your own model weight).
* Run by CPU is default, but if you install **CUDA** appropriately, you can utilize GPU as well.
To check your cuda version, run powershell or other prompt and run this command:
```bash
nvidia-smi
```
You can see this kind of image:
![nvidia-smi](config/images/nvidiasmi.jpg)


Save all the images and bounding boxes (format: class_id, center_x, center_y, width, height).

## Installation
**Windows** with **Anaconda environment** is recommanded.
Make and activate a virtual environment with python 3.10.
```bash
conda create -n IBS_BAP python=3.10
conda activate IBS_BAP
```
Then, copy this repository to your computer.
```bash
git clone https://github.com/spkim8804/IBS_BAP.git
cd IBS_BAP
```
Install libraries.
```bash
pip install -r requirements.txt
```

Run IBS-BAP.
```bash
python IBS_BAP.py
```
