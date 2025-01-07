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
        self.image_list = []
        
        self.is_running = True

    def save_image_bboxes()
        ret, frame = self.cap.read()
        if frame_id in self.bounding_boxes:
            # Make directory for saving files
            save_dir = "./results/" + self.current_filename
            os.makedirs(save_dir, exist_ok=True)
            
            # image save
            image_path = f"{save_dir}/{self.current_filename}_{frame_id:06d}.jpg"
            if not os.path.exists(image_path):
                if mode == "one":
                    cv2.imwrite(image_path, frame)
                    self.progress.emit(f"Saving {self.current_filename}_{frame_id:06d}.jpg")
                elif mode == "all":                    
                    self.image_list.append((frame, frame_id, image_path))
                    
            # bbox save
            bbox_path = f"{save_dir}/{self.current_filename}_{frame_id:06d}.txt"
            with open(bbox_path, "w") as f:
                for bbox in self.bounding_boxes[frame_id]:
                    class_id, x1, y1, x2, y2 = bbox
                    
                    # YOLO 포맷으로 변환
                    x_center = (x1 + x2) / 2 / self.video_width
                    y_center = (y1 + y2) / 2 / self.video_height
                    width = (x2 - x1) / self.video_width
                    height = (y2 - y1) / self.video_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            self.progress.emit(f"Saving {self.current_filename}_{frame_id:06d}.txt")
            
    def run(self):
        if self.mode == "one":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.save_image_bboxes()
        elif self.mode == "all":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for current_frame in range(self.total_frames + 1):
                if not self.is_running: # Interrupted by stop button
                    self.progress.emit("[Info] SaveTask stopped by user.")
                    self.stopped.emit()
                    self.cap.release()
                    return                    
                        
                if len(self.image_list) >= 30 or current_frame == self.total_frames:
                    self.statusBar().showMessage("Saving images..")
                    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                        pool.map(export_frame.export_frame_image, self.image_list)
                    self.image_list = []
                
            self.progress.emit("Saving bounding boxes & images complete!")
            self.cap.release()
            self.finished.emit()

    def stop(self):
        """스레드 실행 중지"""
        self.is_running = False
        self.cap.release()
    