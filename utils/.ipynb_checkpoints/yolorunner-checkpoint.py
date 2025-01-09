import cv2
import torch
import os

from ultralytics import YOLO
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

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
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.is_running = True
        
        self.model = YOLO('./ultralytics/weights/20241216_m_yolov4_data.pt')
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
        for current_frame in range(self.total_frames + 1):
            if not self.is_running: # Interrupted by stop button
                self.progress.emit("[!] Yolo prediction stopped by user.")
                self.result.emit({
                    "bounding_boxes": self.bounding_boxes,
                    "predicted_frame": self.predicted_frame
                })
                self.stopped.emit()
                return
    
            if not self.predicted_frame[current_frame]:
                frame_n = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_n != current_frame:
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, (current_frame / self.fps) * 1000)
                
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = frame.shape
                    
                    results = self.model(frame, verbose=False)
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
                        temp_bbox = (cls, x1, y1, x2, y2)
                
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

        self.result.emit({
            "bounding_boxes": self.bounding_boxes,
            "predicted_frame": self.predicted_frame
        })
        self.progress.emit("Pose estimation complete!")
        self.cap.release()
        self.finished.emit()

    def stop(self):
        """스레드 실행 중지"""
        self.is_running = False
        self.cap.release()
    