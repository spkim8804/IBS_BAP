import cv2
import os
import torch
import multiprocessing
import json
import platform

from ultralytics import YOLO
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

from utils import get_AB_AC_angle

class YoloRunner(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    result = pyqtSignal(dict)
    stopped = pyqtSignal()

    def __init__(self, cap_yolo, predicted_frame, bounding_boxes):
        super().__init__()
        self.cap = cap_yolo
        self.predicted_frame = predicted_frame
        self.bounding_boxes = bounding_boxes
        self.yolo11_model_path = None
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.output = []
        self.model = None

        self.is_running = True

        if platform.system() == "Windows":
            config_path = os.path.join(os.getcwd(), "config\\AVATAR3D_config.json")
        else:
            config_path = os.path.join(os.getcwd(), "config/AVATAR3D_config.json")
        
        with open(config_path, "r") as f:
            config = json.load(f)
            self.yolo11_model_path = config.get("yolo11_model_path")
            
        print(f"modelpath: {self.yolo11_model_path}")
        self.model = YOLO(self.yolo11_model_path)
        self.color_code = {
            "red": (0, 0, 255),
            "orange": (0, 165, 255),
            "yellow": (0, 255, 255),
            "green": (0, 128, 0),
            "blue": (255, 0, 0),
            "skyblue": (235, 206, 135),
            "purple": (128, 0, 128),
            "black": (0, 0, 0),
            "pink": (255, 192, 203)
        }
        self.color_order = ["red", "orange", "yellow", "green", "blue", "skyblue", "purple", "black", "pink"]
        
        #To match darknet (YOLOv4), #4, 5: forelimb / #6, 7: hindlimb
        self.rearrange = [4, 6, 0, 1, 2, 8, 3]
        
        # Five areas from captured frame (x1, y1, x2, y2)
        self.area = [
            [ 1, 1, 1200, 1000],
            [ 1201, 1, 2400, 1000],
            [ 2401, 1, 3600, 1200],
            [ 1, 1001, 1200, 2000],
            [ 1201, 1001, 2400, 2000]
        ]

    def run(self):
        for current_frame in range(1, self.total_frames + 1):
            if not self.is_running: # Interrupted by stop button
                self.progress.emit("[!] Yolo prediction stopped by user.")
                self.result.emit({
                    "bounding_boxes": self.bounding_boxes,
                    "predicted_frame": self.predicted_frame
                })
                self.stopped.emit()
                break
            
            raw_coordinates = [0] * 135
            if not self.predicted_frame[current_frame]:
                if current_frame != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame - 1)
                
                ret, frame = self.cap.read()
                if ret:                    
                    results = self.model(frame, iou=0.4, conf=0.25, verbose=False)
                    self.progress.emit(f"Predict frame {current_frame}/{self.total_frames}...")

                    # Draw bounding box and center
                    check_cls = [[0 for _ in range(10)] for _ in range(10)]
                    raw_coordinates = [0] * 135

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
                        for i in range(5):
                            # Find the ROI (camera 0 to 4)
                            if(center_x > self.area[i][0] and center_x < self.area[i][2] and
                               center_y > self.area[i][1] and center_y < self.area[i][3]):
                                # forehand
                                if(cls == 0 or cls == 1): # 0: fore / 1: hind
                                    if(check_cls[i][self.rearrange[cls]] == 0):
                                        raw_coordinates[i*27 + self.rearrange[cls]*3] = center_x
                                        raw_coordinates[i*27 + 1 + self.rearrange[cls]*3] = center_y
                                        raw_coordinates[i*27 + 2 + self.rearrange[cls]*3] = conf
                                        check_cls[i][self.rearrange[cls]] += 1
                                    elif(check_cls[i][self.rearrange[cls]] == 1):
                                        raw_coordinates[i*27 + self.rearrange[cls]*3 + 3] = center_x
                                        raw_coordinates[i*27 + 1 + self.rearrange[cls]*3 + 3] = center_y
                                        raw_coordinates[i*27 + 2 + self.rearrange[cls]*3 + 3] = conf
                                        check_cls[i][self.rearrange[cls]] += 1
                                elif(check_cls[i][self.rearrange[cls]] == 0): # Rest of body points
                                    raw_coordinates[i*27 + self.rearrange[cls]*3] = center_x
                                    raw_coordinates[i*27 + 1 + self.rearrange[cls]*3] = center_y
                                    raw_coordinates[i*27 + 2 + self.rearrange[cls]*3] = conf
                                    check_cls[i][self.rearrange[cls]] += 1
                    
                    self.predicted_frame[current_frame] = 1
                
            # If feature is not detected, bring a previous value
            # if(current_frame > 1):
            #     for i in range(135):
            #         if(raw_coordinates[i] == 0):
            #             raw_coordinates[i] = self.output[current_frame - 1][i]
        
            # Measure an angle of each limb from body-ass line (4, 5: forelimb / 6, 7: hindlimb / 2: body / 3: ass)
        
            # forelimb (54 is for bottom camera)
            f1_angle = get_AB_AC_angle(raw_coordinates[54 + 3*3], raw_coordinates[54 + 3*3 +1],
                                       raw_coordinates[54 + 2*3], raw_coordinates[54 + 2*3 +1],
                                       raw_coordinates[54 + 4*3], raw_coordinates[54 + 4*3 +1])
        
            f2_angle = get_AB_AC_angle(raw_coordinates[54 + 3*3], raw_coordinates[54 + 3*3 +1],
                                       raw_coordinates[54 + 2*3], raw_coordinates[54 + 2*3 +1],
                                       raw_coordinates[54 + 5*3], raw_coordinates[54 + 5*3 +1])
            
            if(f1_angle < 180 and f2_angle > 180):        
                temp = raw_coordinates[54 + 4*3 : 54 + 4*3 +2]
                raw_coordinates[54 + 4*3: 54 + 4*3 +2] = raw_coordinates[54 +5*3 : 54 +5*3 +2]
                raw_coordinates[54 + 5*3 : 54+ 5*3 +2] = temp
            
            # hindlimb
            h1_angle = get_AB_AC_angle(raw_coordinates[54 + 3*3], raw_coordinates[54 + 3*3+1],
                                       raw_coordinates[54 + 2*3], raw_coordinates[54 + 2*3+1],
                                       raw_coordinates[54 + 6*3], raw_coordinates[54 + 6*3+1])
        
            h2_angle = get_AB_AC_angle(raw_coordinates[54 + 3*3], raw_coordinates[54 + 3*3+1],
                                           raw_coordinates[54 + 2*3], raw_coordinates[54 + 2*3+1],
                                           raw_coordinates[54 + 7*3], raw_coordinates[54 + 7*3+1])
            
            if(h1_angle < 180 and h2_angle > 180):
                temp = raw_coordinates[54 + 6*3 : 54 + 6*3+2]
                raw_coordinates[54 + 6*3 : 54 + 6*3+2] = raw_coordinates[54 +7*3 : 54 + 7*3+2]
                raw_coordinates[54 + 7*3 : 54 + 7*3+2] = temp
    
            # Result accumulation
            self.output.append(raw_coordinates)

        if self.is_running:
            # Save to txt file
            with open('result.txt', 'w') as file:
                for row in self.output:
                    file.write('\t'.join(map(str, row)) + '\n')
            
            self.result.emit({
                "bounding_boxes": self.bounding_boxes,
                "predicted_frame": self.predicted_frame
            })
            self.progress.emit("Pose estimation complete!")
            self.cap.release()
            
        self.finished.emit()

    def stop(self):
        self.is_running = False
        self.cap.release()
        
if __name__ == "__main__":
    multiprocessing.freeze_support()
    