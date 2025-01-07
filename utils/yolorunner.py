import cv2
import torch
import os
import multiprocessing
from ultralytics import YOLO
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

from utils.export_frame import export_frame_image

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
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for current_frame in range(self.total_frames + 1):
            if not self.is_running: # Interrupted by stop button
                self.progress.emit("[!] Yolo prediction stopped by user.")
                self.stopped.emit()
                return
            
            if not self.predicted_frame[current_frame]:
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
        
class YoloSaver(QThread):       
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    stopped = pyqtSignal()
    
    def __init__(self, cap, predicted_frame, bounding_boxes, current_frame, current_filename, mode):
        super().__init__()
        self.cap = cap
        self.predicted_frame = predicted_frame
        self.bounding_boxes = bounding_boxes
        self.current_frame = current_frame
        self.current_filename = current_filename
        self.mode = mode
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.image_list = []
        
        self.is_running = True

    def save_image_bboxes(self):
        ret, frame = self.cap.read()
        if self.current_frame in self.bounding_boxes:
            # Make directory for saving files
            save_dir = "./results/" + self.current_filename
            os.makedirs(save_dir, exist_ok=True)
            
            # image save
            image_path = f"{save_dir}/{self.current_filename}_{self.current_frame:06d}.jpg"
            if not os.path.exists(image_path):
                if self.mode == "one":
                    cv2.imwrite(image_path, frame)
                    self.progress.emit(f"Saving {self.current_filename}_{self.current_frame:06d}.jpg")
                elif self.mode == "all":                    
                    self.image_list.append((frame, self.current_frame, image_path))
                    
            # bbox save
            bbox_path = f"{save_dir}/{self.current_filename}_{self.current_frame:06d}.txt"
            with open(bbox_path, "w") as f:
                for bbox in self.bounding_boxes[self.current_frame]:
                    class_id, x1, y1, x2, y2 = bbox
                    
                    # YOLO 포맷으로 변환
                    x_center = (x1 + x2) / 2 / self.video_width
                    y_center = (y1 + y2) / 2 / self.video_height
                    width = (x2 - x1) / self.video_width
                    height = (y2 - y1) / self.video_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            self.progress.emit(f"Saving {self.current_filename}_{self.current_frame:06d}.txt")
            
    def run(self):
        if self.mode == "one":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.save_image_bboxes()
        elif self.mode == "all":
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for current_frame in range(self.total_frames + 1):
                if not self.is_running: # Interrupted by stop button
                    self.progress.emit("[!] Saving process stopped by user.")
                    self.stopped.emit()
                    self.cap.release()
                    return                    
                self.current_frame = current_frame
                self.save_image_bboxes()
                
                if len(self.image_list) >= 30 or current_frame == self.total_frames:
                    self.progress.emit(" Saving images...")
                    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                        pool.map(export_frame_image, self.image_list)
                        
                    self.image_list = []
                
            self.progress.emit("Saving bounding boxes & images complete!")
            self.cap.release()
            self.finished.emit()

    def stop(self):
        """스레드 실행 중지"""
        self.is_running = False
        self.cap.release()
    