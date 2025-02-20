# IBS Behavior Analysis Program (IBS-BAP)
* IBS-BAP is a project aimed at creating a powerful and convenient behavior analysis program.
* The ultimate goal is to enable one-step behavior analysis, from classification to analysis, directly from images or videos.
* Currently, it supports pose estimation using YOLO11, as well as visualization and management of bounding box label data.

## Updates
* 2025/02/04: Confirmed that IBS-BAP is working well on macOS.
* 2025/02/13: Pose visualizer is now available in "Visualizer" menu (Windows and macOS).

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
conda create -n IBS_BAP python=3.10 -y
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

## GUI composition
![composition](config/images/IBS_BAP_Composition.jpg)
### File - Open files (Ctrl + O)
You can open image (".jpg", ".png") or video (".mp4") files and filelist is on the bottom center.
If you have annotation data (".txt" and ".json" for for image and mp4 file, respectively) with same filename, the program will automatically load corresponding annotation file.
### Image
You can see image or frame in the left panel.
* If you **drag with mouse left click**, you can make bounding box to annotate with class indicated on the top right panel ("fore" in this example). You can change class by click scroll-down or number in the keyboard (0-9 maximum).
* To remove the bounding box, you can simply **mouse right-click** nearby your target box.
* Zoom In/Zoom out: Using **mouse wheel** to magnify image for accurate bounding box validation.
### Class selection
* You can choose class for annotation. **Default is AVATAR3D configuration** (7 classes: fore, hind, nose, head, ass, tail, torso).
* If you want to annotate other classes, you can edit "class" section in "./config/AVATAR3D_config.json".
### Bounding boxes list
* You can find current bounding boxes. If you click the item then it will be highlighted as **red box** in the image.
### Controller
* Play: ▶ (Space) / ⏸(Space) button
* Move frame: "< (d)" previous frame / "> (f)" next frame
### Buttons
* Save current BBox (Ctrl + S): Save current image and bounding boxes
* Save all BBox: Save image and bounding boxes for frames which has bounding boxes
* Run yolo11: Predict keypoint using yolo11. If you have NVIDIA GPU and properly setup with **CUDA driver** and **pytorch cudatoolkit**, you can use GPU for prediction.
* Stop task: You can terminate ongoing task (ex. yolo11 prediction or saving process)
