import sys
import cv2
import numpy as np
import json
import random
import os
import multiprocessing

from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, 
    QWidget, QFileDialog, QSlider, QHBoxLayout, QListWidget, QListWidgetItem,
    QComboBox, QScrollArea, QMenuBar, QAction, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import QTimer, Qt, QPoint

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBS Behavior analysis program")
        self.setGeometry(100, 100, 1200, 700)

        # Menu bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        # Status bar
        self.status_label = QLabel("IBS_BAP by Sunpil Kim", self)
        self.statusBar().addPermanentWidget(self.status_label)

        open_action = QAction("Open Video", self)
        open_action.setShortcut("Ctrl+O")  # 단축키 설정
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # ======== UI 구성 ========
        self.video_player_width = 900
        self.video_player_height = 600
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFixedSize(self.video_player_width, self.video_player_height)
        
        self.video_label = QLabel(self)
        self.video_label.resize(self.video_player_width, self.video_player_height)  # 기본 크기
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        self.scroll_area.setWidget(self.video_label)

        # 재생 슬라이더
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setRange(0, 100)
        self.playback_slider.setValue(0)
        self.playback_slider.setToolTip("Playback Progress")
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFixedWidth(120)

        # 드롭다운 (Class 선택)
        self.class_selector = QComboBox()
        
        # 버튼
        self.play_button = QPushButton("▶(Space)") # "⏸ Pause"
        self.previous_frame_button = QPushButton("< (d)")
        self.next_frame_button = QPushButton("> (f)")
        self.save_box_button = QPushButton("💾 Save current BBox")
        self.save_all_button = QPushButton("💾 Save All BBox")
        self.delete_box_button = QPushButton("❌ Delete Selected Box")

        # Bounding Box 리스트
        self.bbox_list = QListWidget()
        
        # 비디오 재생 목록
        self.playlist = QListWidget()

        # 레이아웃 설정
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.previous_frame_button)
        controls_layout.addWidget(self.next_frame_button)
        controls_layout.addWidget(self.class_selector)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.playlist)

        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.scroll_area)
        video_layout.addWidget(self.playback_slider)
        video_layout.addLayout(controls_layout)

        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.bbox_list)
        bbox_layout.addWidget(self.save_box_button)
        bbox_layout.addWidget(self.save_all_button)
        bbox_layout.addWidget(self.delete_box_button)
        
        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(bbox_layout, 1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ======== 이벤트 핸들러 연결 ========
        self.play_button.clicked.connect(self.play_button_click)
        self.previous_frame_button.clicked.connect(self.move_to_previous_frame)
        self.next_frame_button.clicked.connect(self.move_to_next_frame)
        self.save_box_button.clicked.connect(self.save_current_frame_bboxes)
        self.save_all_button.clicked.connect(self.save_all_bboxes)
        self.delete_box_button.clicked.connect(self.delete_selected_box)
        self.playlist.itemClicked.connect(self.play_selected_video)
        self.playback_slider.sliderMoved.connect(self.seek_video)
        self.class_selector.currentIndexChanged.connect(self.class_selected)

        # ==== Video configuration ======
        self.is_playing = False
        self.cap = None
        self.video_files = []
        self.total_frames = 0
        self.current_frame = 0
        
        # ==== Timer setting ========
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_frame)
        self.save_timer = QTimer(self)
        self.save_timer.timeout.connect(lambda: self.save_current_frame(self.current_frame))
        

        # ======== Bounding Box ========
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.bounding_boxes = {}
        self.current_filename = None

        # JSON 설정 불러오기
        self.classes = []
        self.current_class_id = 0
        self.load_classes()
        
        # ======== 키보드 이벤트 ========
        self.setFocusPolicy(Qt.StrongFocus)

        # 확대/축소 관련 변수
        self.h_scroll = self.scroll_area.horizontalScrollBar().value()
        self.v_scroll = self.scroll_area.verticalScrollBar().value()
        self.zoom_scale = 1.0
        self.original_scale = 1.0

        ###### YOLO setting ############
        self.model = YOLO('./ultralytics/weights/20241216_m_yolov4_data.pt')
        # Color code setting for center-point
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
        self.predicted_frame = []

    # ======== 동영상 열기 ========
    def open_video(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Video Files", "", "Video Files (*.mp4 *.avi *.mkv)")
        if files:
            self.video_files.extend(files)
            self.playlist.addItems(files)

    def play_selected_video(self, item: QListWidgetItem):
        if self.cap:
            reply = QMessageBox.warning(
                self, 
                "Warning", 
                "You will lose current works. Do you want to continue?", 
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            else:
                self.cap.release()

        file_path = item.text()
        self.current_filename, ext = os.path.splitext(file_path.split("/")[-1])
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("[Error] Cannot open video file:", file_path)
            return
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_scale = min(self.video_player_width / self.video_width,
                                  self.video_player_height / self.video_height)
        self.zoom_scale = self.original_scale
        self.playback_slider.setRange(0, self.total_frames)
        
        self.predicted_frame = [0]*(self.total_frames + 1)
        
        self.update_frame_label()
        self.refresh_frame()

    # ======== Video controller ============
    def play_button_click(self):
        if self.cap:
            if self.is_playing == True:
                self.pause_video()
            else:
                self.play_video()
                
    def play_video(self):
        self.is_playing = True
        self.play_button.setText("⏸(Space)")
        self.video_timer.start(20)

    def pause_video(self):
        self.is_playing = False
        self.play_button.setText("▶(Space)")
        self.video_timer.stop()

    def stop_video(self):
        if self.cap:
            self.is_playing = False
            self.video_timer.stop()
            self.cap.release()
            self.video_label.clear()
            self.playback_slider.setValue(0)

    # ======== JSON 클래스 로드 ========
    def load_classes(self):
        try:
            with open("./config/AVATAR3D_config.json", "r") as f:
                config = json.load(f)
                self.classes = config.get("class", [])
                self.class_selector.clear()  # 기존 항목 제거
                self.class_selector.addItems(self.classes)

                # 클래스별 고유 색상 할당
                self.class_colors = self.generate_class_colors(len(self.classes))
        except Exception as e:
            print(f"[Error] Failed to load classes: {e}")

    # ======== Class_id selection ========
    def class_selected(self, index):
        self.current_class_id = index
        
    def generate_class_colors(self, num_classes):
        random.seed(42)  # 재현 가능한 색상 생성
        colors = []
        for _ in range(num_classes):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return colors
    
    # ======== Update frame ========
    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:                               
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame.shape

                ###### YOLO part ###############################
                # Run YOLOv8 inference on the frame
                if not self.predicted_frame[self.current_frame]:
                    results = self.model(frame, verbose=False)
                    self.predicted_frame[self.current_frame] = 1
                    
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
                
                        if self.current_frame not in self.bounding_boxes:
                            self.bounding_boxes[self.current_frame] = []
                        self.bounding_boxes[self.current_frame].append(temp_bbox)
                
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

                ###### YOLO part ###############################
                
                # Ongoing Bounding Box
                if self.start_point and self.is_drawing == True:
                    start_x = int((self.start_point[0] + self.h_scroll) / self.zoom_scale)
                    start_y = int((self.start_point[1] + self.v_scroll) / self.zoom_scale)
                    end_x = int((self.end_point[0] + self.h_scroll) / self.zoom_scale)
                    end_y = int((self.end_point[1] + self.v_scroll) / self.zoom_scale)
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)

                # Bounding Box in current frame
                if self.current_frame in self.bounding_boxes:
                    for bbox in self.bounding_boxes[self.current_frame]:
                        class_id, x1, y1, x2, y2 = bbox
                        color = self.class_colors[class_id % len(self.class_colors)]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
                self.video_label.resize(int(self.video_width * self.zoom_scale), int(self.video_height * self.zoom_scale))
                self.update_frame_label()
                self.update_bbox_list()
                self.playback_slider.setValue(self.current_frame)
            else:
                self.pause_video()

    # ======== video slider controller ========
    def seek_video(self, position):
        if self.cap:
            self.video_timer.stop()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            # self.refresh_frame()

    # ======== Bounding Box 이벤트 ========
    def mousePressEvent(self, event):
        self.get_scroll_position()
        if (event.button() == Qt.LeftButton and
            0 <= event.x() - 10 <= self.video_player_width and
            0 <= event.y() - 10 - self.menuBar().height() <= self.video_player_height):
            
            self.pause_video()
            self.start_point = (event.x() - 10, event.y() - 10 - self.menuBar().height())
            self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.pause_video()
            self.end_point = (min(event.x() - 10, self.video_player_width),
                              min(event.y() - 10 - self.menuBar().height(), self.video_player_height))
            self.refresh_frame()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.is_drawing:
            self.end_point = (min(event.x() - 10, self.video_player_width),
                              min(event.y() - 10 - self.menuBar().height(), self.video_player_height))
            self.is_drawing = False

            # Filter just click event, not bounding box drawing
            if self.start_point[0] == self.end_point[0] or self.start_point[1] == self.end_point[1]:
                return

            x1 = int((self.start_point[0] + self.h_scroll) / self.zoom_scale)
            y1 = int((self.start_point[1] + self.v_scroll) / self.zoom_scale)
            x2 = int((self.end_point[0] + self.h_scroll) / self.zoom_scale)
            y2 = int((self.end_point[1] + self.v_scroll) / self.zoom_scale)
            x1y1 = (min(x1, x2), min(y1, y2))
            x2y2 = (max(x1, x2), max(y1, y2))
            temp_bbox = (self.current_class_id, *x1y1, *x2y2)
            
            if self.current_frame not in self.bounding_boxes:
                self.bounding_boxes[self.current_frame] = []
            self.bounding_boxes[self.current_frame].append(temp_bbox)

            self.update_bbox_list()

            self.refresh_frame()
            self.start_point = None
            self.end_point = None
        
    # ======== 확대/축소 ========
    def wheelEvent(self, event):
        # Scale change only when Ctrl key is pressed
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_scale = min(self.original_scale * 5.0, self.zoom_scale + 0.1)
            else:
                self.zoom_scale = max(self.original_scale, self.zoom_scale - 0.1)
            
            self.get_scroll_position()
            self.refresh_frame()
        else:
            super().wheelEvent(event)

    def get_scroll_position(self):
        self.h_scroll = self.scroll_area.horizontalScrollBar().value()
        self.v_scroll = self.scroll_area.verticalScrollBar().value()
        
    # ======== Bounding Box save or delete ========
    def update_bbox_list(self):
        """현재 프레임의 Bounding Box를 리스트에 표시"""
        self.bbox_list.clear()        
        if self.current_frame in self.bounding_boxes:
            for bbox in self.bounding_boxes[self.current_frame]:
                class_id, x1, y1, x2, y2 = bbox
                self.bbox_list.addItem(
                    f"Class: {self.classes[class_id]} | ({x1}, {y1}) -> ({x2}, {y2})"
                )

    def save_current_frame(self, frame_id):
        if not self.cap:
            print("[Error] No video loaded.")
            return
        
        if frame_id in self.bounding_boxes:
            # Make directory for saving files
            save_dir = "./results/" + self.current_filename
            os.makedirs(save_dir, exist_ok=True)
            
            # image save
            image_path = f"{save_dir}/{self.current_filename}_{frame_id:06d}.jpg"
            if not os.path.exists(image_path):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"[Error] Failed to read frame {frame_id}.")
                    return
                
                cv2.imwrite(image_path, frame)
                self.statusBar().showMessage(f"{self.current_filename}_{frame_id:06d}.jpg saved")
    
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
    
            self.statusBar().showMessage(f"{self.current_filename}_{frame_id:06d}.txt saved")

        # For save all bboxes
        self.current_frame += 1
        if(self.current_frame == self.total_frames):
            self.statusBar().showMessage("Saving complete!", 2000)
            self.reset_statusBar()
            self.save_timer.stop()
            self.current_frame = 0
        
    def save_current_frame_bboxes(self):
        self.save_current_frame(self.current_frame)
        self.current_frame -= 1

    def save_all_bboxes(self):
        self.current_frame = 0
        self.save_timer.start(1)
    
    def delete_selected_box(self):
        selected = self.bbox_list.currentRow()
        if selected >= 0:
            self.bounding_boxes[self.current_frame].pop(selected)
            self.bbox_list.takeItem(selected)
            self.refresh_frame()
    
    def keyPressEvent(self, event):
        print(event.key())
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl 키와 함께 방향키 입력 시 프레임 이동
            if event.key() == Qt.Key_Right and self.is_playing == False:
                self.move_to_next_frame()
                self.refresh_frame()
            elif event.key() == Qt.Key_Left and self.is_playing == False:
                self.move_to_previous_frame()
                self.refresh_frame()
        else:
            if event.key() == Qt.Key_Space:
                self.play_button_click()
            elif 48 <= event.key() < 48 + len(self.classes):
                self.current_class_id = event.key() - 48
                self.class_selector.setCurrentIndex(self.current_class_id)
            elif event.key() == Qt.Key_Delete:
                self.delete_selected_box()
            elif event.key() == 68: # d key
                self.move_to_previous_frame()
            elif event.key() == 70: # f key
                self.move_to_next_frame()

    def move_to_next_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, min(self.current_frame + 1, self.total_frames))
            self.refresh_frame()

    def move_to_previous_frame(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(self.current_frame - 1, -1))
            self.refresh_frame()
            
    def refresh_frame(self):
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(self.current_frame - 1, 0))
        self.update_frame()
        
    def update_frame_label(self):
        self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_label.setText(f"Frame: {self.current_frame}/{self.total_frames}")
    
    def reset_statusBar(self):
        self.status_label.setText("IBS_BAP by Sunpil Kim")
        
if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    player = VideoPlayer()
    player.show()

    try:
        app.exec_()
    except SystemExit:
        print("[Info] PyQt5 Application exited cleanly.")
    finally:
        app.quit()
        del app
        print("[Info] QApplication resources have been cleaned up.")