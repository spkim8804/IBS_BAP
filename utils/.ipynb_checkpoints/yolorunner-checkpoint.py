import cv2
import os
import torch
import multiprocessing
import json
import platform

from ultralytics import YOLO
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from utils import get_AB_AC_angle

class YoloRunner(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    stopped = pyqtSignal()

    def __init__(self, cap, predicted_frame, bounding_boxes, config_path):
        super().__init__()
        self.cap = cap
        self.predicted_frame = predicted_frame
        self.bounding_boxes = bounding_boxes
        self.config_path = config_path
        self.camera_info = []
        self.area = []
        self.keypoint = 0
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output = []
        self.model = None

        self.is_running = True
        
        with open(self.config_path, "r") as f:
            try:
                self.cfg = json.load(f)
                
                if self.cfg["camera"]["cnt"] > 1:
                    self.area = self.cfg["camera"]["roi"]
                else: 
                    self.area = [[0, 0, self.video_width, self.video_height]]
            except:
                QMessageBox.warning(self, "QMessageBox", "Please check config file!")

        self.model = YOLO(self.cfg["yolo11_model_path"])
        
        #To match darknet (YOLOv4), #4, 5: forelimb / #6, 7: hindlimb
        # 0: "nose", 1: "head", 2: "ass", 3: "chest", 4: "rleg", 5: "lleg", 6: "rarm", 7: "larm", 8: "tail"
        # YOLO label: 0: "fore", 1: "hind", 2: "nose", 3: "head", 4: "ass", 5: "tail", 6: "torso"
        self.rearrange = [6, 4, 0, 1, 2, 8, 3]

    def export_results(self):
        self.result.emit({
            "bounding_boxes": self.bounding_boxes,
            "predicted_frame": self.predicted_frame,
            "raw_coordinates": self.output
        })
        # Save to txt file
        # with open('result.txt', 'w') as file:
        #     for row in self.output:
        #         file.write('\t'.join(map(str, row)) + '\n')
        
        self.stopped.emit()
        self.cap.release()
        
    def run(self):
        raw_coordinates = [0] * (3 * self.cfg["keypoint"]["cnt"] * self.cfg["camera"]["cnt"])
        for current_frame in range(1, self.total_frames + 1):
            if current_frame == 1:
                self.output.append(raw_coordinates)
                
            if not self.is_running: # Interrupted by stop button
                self.progress.emit("[!] Yolo prediction stopped by user.")
                self.export_results()
                break

            if not self.predicted_frame[current_frame]:
                if current_frame != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                
                ret, frame = self.cap.read()

                if ret:
                    results = self.model(frame, iou=0.4, conf=0.25, verbose=False)
                    self.progress.emit(f"Predict frame: {current_frame}/{self.total_frames}...")

                    # Draw bounding box and center
                    check_cls = [[0 for _ in range(10)] for _ in range(10)]
                    raw_coordinates = [0] * (3 * self.cfg["keypoint"]["cnt"] * self.cfg["camera"]["cnt"])

                    for det in results[0].boxes:
                        # det: [x1, y1, x2, y2, conf, cls]
                        xy = det.xyxy.tolist()[0]
                        conf, cls = det.conf.item(), int(det.cls.item())
                        # cls definition
                        # 0: fore / 1: hind / 2: nose / 3: head / 4: ass / 5: tail / 6: torso
            
                        # Calculate the center of the bounding box
                        center_x, center_y = (int(xy[0]) + int(xy[2])) // 2, (int(xy[1]) + int(xy[3])) // 2
                        x1 = min(int(xy[0]), int(xy[2]))
                        x2 = max(int(xy[0]), int(xy[2]))
                        y1 = min(int(xy[1]), int(xy[3]))
                        y2 = max(int(xy[1]), int(xy[3]))
                        temp_bbox = [cls, x1, y1, x2, y2]
                
                        if current_frame not in self.bounding_boxes:
                            self.bounding_boxes[current_frame] = []
                        self.bounding_boxes[current_frame].append(temp_bbox)
                
                        # Get only one xy pair (highest conf value) in each camera view
                        for i in range(int(self.cfg["camera"]["cnt"])):
                            # Find the ROI (camera 0 to 4)
                            if(center_x > self.area[i][0] and center_x < self.area[i][2] and
                               center_y > self.area[i][1] and center_y < self.area[i][3]):
                                # forehand
                                if(cls == 0 or cls == 1): # 0: fore / 1: hind
                                    if(check_cls[i][self.rearrange[cls]] == 0):
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + self.rearrange[cls]*3] = center_x
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 1 + self.rearrange[cls]*3] = center_y
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 2 + self.rearrange[cls]*3] = conf
                                        check_cls[i][self.rearrange[cls]] += 1
                                    elif(check_cls[i][self.rearrange[cls]] == 1):
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + self.rearrange[cls]*3 + 3] = center_x
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 1 + self.rearrange[cls]*3 + 3] = center_y
                                        raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 2 + self.rearrange[cls]*3 + 3] = conf
                                        check_cls[i][self.rearrange[cls]] += 1
                                elif(check_cls[i][self.rearrange[cls]] == 0): # Rest of body points
                                    raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + self.rearrange[cls]*3] = center_x
                                    raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 1 + self.rearrange[cls]*3] = center_y
                                    raw_coordinates[i*3*int(self.cfg["keypoint"]["cnt"]) + 2 + self.rearrange[cls]*3] = conf
                                    check_cls[i][self.rearrange[cls]] += 1
                    
                    self.predicted_frame[current_frame] = 1
                
            # If feature is not detected, bring a previous value
            if(current_frame > 1):
                for i in range(3 * self.cfg["keypoint"]["cnt"] * self.cfg["camera"]["cnt"]):
                    if(raw_coordinates[i] == 0):
                        raw_coordinates[i] = self.output[current_frame - 1][i]
        
            # Measure an angle of each limb from body-ass line (4, 5: forelimb / 6, 7: hindlimb / 2: body / 3: ass)
        
            # forelimb (Based on the bottom camera)
            bottom_idx = self.cfg["camera"]["bottom_camera"] * 3 * self.cfg["keypoint"]["cnt"]
            f1_angle = get_AB_AC_angle(raw_coordinates[bottom_idx + 3*3], raw_coordinates[bottom_idx + 3*3 +1],
                                       raw_coordinates[bottom_idx + 2*3], raw_coordinates[bottom_idx + 2*3 +1],
                                       raw_coordinates[bottom_idx + 4*3], raw_coordinates[bottom_idx + 4*3 +1])
        
            f2_angle = get_AB_AC_angle(raw_coordinates[bottom_idx + 3*3], raw_coordinates[bottom_idx + 3*3 +1],
                                       raw_coordinates[bottom_idx + 2*3], raw_coordinates[bottom_idx + 2*3 +1],
                                       raw_coordinates[bottom_idx + 5*3], raw_coordinates[bottom_idx + 5*3 +1])
            
            if(f1_angle < 180 and f2_angle > 180):
                temp = raw_coordinates[bottom_idx + 4*3 : bottom_idx + 4*3 +2]
                raw_coordinates[bottom_idx + 4*3: bottom_idx + 4*3 +2] = raw_coordinates[bottom_idx +5*3 : bottom_idx +5*3 +2]
                raw_coordinates[bottom_idx + 5*3 : bottom_idx+ 5*3 +2] = temp
            
            # hindlimb
            h1_angle = get_AB_AC_angle(raw_coordinates[bottom_idx + 3*3], raw_coordinates[bottom_idx + 3*3+1],
                                       raw_coordinates[bottom_idx + 2*3], raw_coordinates[bottom_idx + 2*3+1],
                                       raw_coordinates[bottom_idx + 6*3], raw_coordinates[bottom_idx + 6*3+1])
        
            h2_angle = get_AB_AC_angle(raw_coordinates[bottom_idx + 3*3], raw_coordinates[bottom_idx + 3*3+1],
                                           raw_coordinates[bottom_idx + 2*3], raw_coordinates[bottom_idx + 2*3+1],
                                           raw_coordinates[bottom_idx + 7*3], raw_coordinates[bottom_idx + 7*3+1])
            
            if(h1_angle < 180 and h2_angle > 180):
                temp = raw_coordinates[bottom_idx + 6*3 : bottom_idx + 6*3+2]
                raw_coordinates[bottom_idx + 6*3 : bottom_idx + 6*3+2] = raw_coordinates[bottom_idx +7*3 : bottom_idx + 7*3+2]
                raw_coordinates[bottom_idx + 7*3 : bottom_idx + 7*3+2] = temp
    
            # Result accumulation
            self.output.append(raw_coordinates)

        if self.is_running:
            self.progress.emit("Pose estimation complete!")
            self.export_results()
            
        self.finished.emit()

    def stop(self):
        self.is_running = False
        self.cap.release()
        
if __name__ == "__main__":
    multiprocessing.freeze_support()
    