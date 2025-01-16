import cv2
import os
import multiprocessing

from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

from utils.export_frame import export_frame_image

class YoloSaver(QThread):       
    progress = pyqtSignal(str)
    finished = pyqtSignal()
    stopped = pyqtSignal()
    
    def __init__(self, cap, predicted_frame, bounding_boxes, current_frame_n, current_videopath, mode):
        super().__init__()
        self.cap = cap
        self.predicted_frame = predicted_frame
        self.bounding_boxes = bounding_boxes
        self.current_frame_n = current_frame_n
        self.current_videopath = current_videopath
        self.mode = mode
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.image_list = []
        
        self.is_running = True

    def save_image_bboxes(self):
        if self.current_frame_n in self.bounding_boxes:
            if self.current_frame_n != int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_n - 1)
            
            ret, frame = self.cap.read()
            # Make directory for saving files
            directory, filename = os.path.split(self.current_videopath)
            filename, ext = os.path.splitext(filename)

            if ext == ".mp4":
                save_dir = f"{directory}/{filename}"
                os.makedirs(save_dir, exist_ok=True)
            elif ext == ".jpg" or ".png":
                save_dir = directory
            
            # image save
            if ext == ".mp4":
                image_path = f"{save_dir}/{filename}_{self.current_frame_n:06d}.jpg"
                if not os.path.exists(image_path):
                    if self.mode == "one":
                        cv2.imwrite(image_path, frame)
                        self.progress.emit(f"Saving {filename}_{self.current_frame_n:06d}.jpg")
                    elif self.mode == "all":                    
                        self.image_list.append((frame, self.current_frame_n, image_path))
                    
            # bbox save
            if ext == ".mp4":
                bbox_path = f"{save_dir}/{filename}_{self.current_frame_n:06d}.txt"
            elif ext == ".jpg" or ".png":
                bbox_path = f"{save_dir}/{filename}.txt"

            with open(bbox_path, "w") as f:
                for bbox in self.bounding_boxes[self.current_frame_n]:
                    class_id, x1, y1, x2, y2 = bbox
                    
                    # YOLO 포맷으로 변환
                    x_center = (x1 + x2) / 2 / self.video_width
                    y_center = (y1 + y2) / 2 / self.video_height
                    width = (x2 - x1) / self.video_width
                    height = (y2 - y1) / self.video_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            self.progress.emit(f"Saving {filename}_{self.current_frame_n:06d}.txt")
    
    def run(self):
        if self.mode == "one":
            self.save_image_bboxes()
        elif self.mode == "all":
            for current_frame in range(1, self.total_frames + 1):
                if not self.is_running: # Interrupted by stop button
                    self.progress.emit("[!] Saving process stopped by user.")
                    self.stopped.emit()
                    self.cap.release()
                    return
                self.current_frame_n = current_frame
                self.save_image_bboxes()
                
                if len(self.image_list) >= 30 or self.current_frame_n == self.total_frames:
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

if __name__ == "__main__":
    freeze_support()
        