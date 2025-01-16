# 2024-01-15 ver
# Index: File management, Video controller, Load configuration, Yolo class, Yolo running
#        Update frame, Mouse control, Keyboard control, BBox management, 
#        Task management, Video conversion, Main program
import sys
import cv2
import numpy as np
import json
import random
import os
import platform
import multiprocessing

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QSlider, QHBoxLayout, QListWidget, QListWidgetItem,
    QComboBox, QScrollArea, QMenuBar, QAction, QMessageBox, QDialog, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

from utils import (
    get_screen_resolution, CheckVideoFormat, ConvertVideoToIframe, check_video_format,
    YoloRunner, YoloSaver, calc_min_distance
)
        
class IBS_BAP(QMainWindow):
    def __init__(self):
        super().__init__()
        self.converter_window = None

        # Window size setting depending on the system resolution
        # Default: HD resolution (1280 x 720)
        self.video_player_width = 1280
        self.video_player_height = 720
        os_name = platform.system()
        print(f"Current OS: {os_name}")
        if os_name == "Windows":
            self.video_player_width, self.video_player_height = get_screen_resolution()
        
        self.video_player_width = int(self.video_player_width * 60 / 100)
        self.video_player_height = int(self.video_player_height * 60 / 100)
        
        self.setWindowTitle("IBS Behavior Analysis Program (BAP)")
        self.setGeometry(10, 40, int(self.video_player_width * 1.2), int(self.video_player_height * 1.2))
        #### Menu bar ########################
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        
        open_action = QAction("Open Files", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_files)
        file_menu.addAction(open_action)

        save_current_action = QAction("Save Current BBoxes", self)
        save_current_action.setShortcut("Ctrl+S")
        save_current_action.triggered.connect(lambda: self.save_files("one"))
        file_menu.addAction(save_current_action)

        save_all_action = QAction("Save All BBoxes", self)
        save_all_action.triggered.connect(lambda: self.save_files("all"))
        file_menu.addAction(save_all_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu_bar.addMenu("&Edit")
        refresh_action = QAction("Refresh current frame", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_frame)
        edit_menu.addAction(refresh_action)
        
        ### Status bar ########################
        self.status_label = QLabel("IBS_BAP by Sunpil Kim", self)
        self.statusBar().addPermanentWidget(self.status_label)
        
        # ======== UI interface ========
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFixedSize(self.video_player_width + 10, self.video_player_height + 10)
        
        self.video_label = QLabel(self)
        self.video_label.resize(self.video_player_width, self.video_player_height)  # 기본 크기
        self.video_label.setScaledContents(True)
        self.video_label.setStyleSheet("background-color: black;")
        self.scroll_area.setWidget(self.video_label)

        # playback slider
        self.playback_slider = QSlider(Qt.Horizontal)
        self.playback_slider.setRange(0, 100)
        self.playback_slider.setValue(0)
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFixedWidth(120)
        self.class_label = QLabel("Class selection")
        self.bbox_list_label = QLabel("Bounding boxes list")

        # Class selector (dropdown)
        self.class_selector = QComboBox()
        
        # 버튼
        self.play_button = QPushButton("▶(Space)") # "⏸ Pause"
        self.previous_frame_button = QPushButton("< (d)")
        self.next_frame_button = QPushButton("> (f)")
        self.save_box_button = QPushButton("💾 Save current BBox")
        self.save_all_button = QPushButton("💾 Save All BBox")
        # self.delete_box_button = QPushButton("❌ Delete Selected Box")
        self.yolo_button = QPushButton("Run yolo11")
        self.stop_button = QPushButton("🛑 Stop Task")

        # Bounding Box list
        self.bbox_list = QListWidget()
        
        # Playlist
        self.playlist = QListWidget()

        # GUI layout
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.previous_frame_button)
        controls_layout.addWidget(self.next_frame_button)
        controls_layout.addWidget(self.frame_label)
        controls_layout.addWidget(self.playlist)

        main_layout = QHBoxLayout()
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.scroll_area)
        video_layout.addWidget(self.playback_slider)
        video_layout.addLayout(controls_layout)

        bbox_layout = QVBoxLayout()
        bbox_layout.addWidget(self.class_label)
        bbox_layout.addWidget(self.class_selector)
        bbox_layout.addWidget(self.bbox_list_label)
        bbox_layout.addWidget(self.bbox_list)
        bbox_layout.addWidget(self.save_box_button)
        bbox_layout.addWidget(self.save_all_button)
        # bbox_layout.addWidget(self.delete_box_button)
        bbox_layout.addWidget(self.yolo_button)
        bbox_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(video_layout, 3)
        main_layout.addLayout(bbox_layout, 1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ==== Video configuration ======
        self.is_playing = False
        self.cap = None
        self.cap_yolo = None
        self.cap_save = None
        self.filelist = []
        self.total_frames = 0
        self.fps = 0
        self.current_frame = None
        self.current_frame_n = 0
        self.current_filename = None
        self.current_ext= None
        self.current_directory = None
        self.current_videopath = None
        
        # ==== Timer setting ========
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_frame)

        # ======== Bounding Box ========
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.bounding_boxes = {}

        ###### YOLO setting ############
        self.predicted_frame = []
        
        # JSON setting
        self.classes = []
        self.current_class_id = 0
        self.load_config()
        
        # ======== Keyboard event ========
        self.setFocusPolicy(Qt.StrongFocus)

        # Zoom in/out
        self.h_scroll = self.scroll_area.horizontalScrollBar().value()
        self.v_scroll = self.scroll_area.verticalScrollBar().value()
        self.zoom_scale = 1.0
        self.original_scale = 1.0

        # ======== Event handler connect ========
        self.play_button.clicked.connect(self.play_button_click)
        self.previous_frame_button.clicked.connect(self.move_to_previous_frame)
        self.next_frame_button.clicked.connect(self.move_to_next_frame)
        self.save_box_button.clicked.connect(lambda: self.save_files("one"))
        self.save_all_button.clicked.connect(lambda: self.save_files("all"))
        # self.delete_box_button.clicked.connect(self.delete_selected_box)
        self.playlist.itemClicked.connect(self.play_selected_file)
        self.bbox_list.itemClicked.connect(self.selected_bbox)
        self.playback_slider.sliderPressed.connect(self.slider_pressed)
        self.playback_slider.sliderReleased.connect(self.slider_released)
        self.class_selector.currentIndexChanged.connect(self.class_selected)
        self.yolo_button.clicked.connect(self.run_yolo)
        self.stop_button.clicked.connect(self.stop_save_task)
    
    ### File management start #######################
    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Files (*.mp4 *.jpg *.png)")
        if files:
            self.filelist.extend(files)
            self.playlist.addItems(files)

    def load_bounding_boxes(self, bbox_path, video_width, video_height):
        bounding_boxes = []
        if os.path.exists(bbox_path): # 파일이 존재하는지 확인
            with open(bbox_path, "r") as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, width, height = map(float, line.split())
                        class_id = int(class_id) # class_id 는 정수형으로 변환
    
                        # 원래 좌표로 변환
                        x1 = int((x_center - width / 2) * video_width)
                        y1 = int((y_center - height / 2) * video_height)
                        x2 = int((x_center + width / 2) * video_width)
                        y2 = int((y_center + height / 2) * video_height)
    
                        bounding_boxes.append([class_id, x1, y1, x2, y2])
                    except ValueError:
                        print(f"경고: 잘못된 형식의 라인 발견: {line.strip()}") # 잘못된 형식의 라인 처리
        
        return bounding_boxes
    
    def play_selected_file(self, item: QListWidgetItem):
        self.pause_video()
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
                if self.cap:
                    self.cap.release()
                if self.cap_yolo:
                    self.cap_yolo.release()

        file_path = item.text()
        directory, filename = os.path.split(file_path)
        filename, ext = os.path.splitext(filename)
        file_removed = False
        
        if ext in [".jpg", ".png"]: # Treat as one frame video
            temp_frame = cv2.imread(file_path)
            height, width, _ = temp_frame.shape
            temp_path = "./utils/temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_writer = cv2.VideoWriter(temp_path, fourcc, 1, (width, height))
            temp_writer.write(temp_frame)
            temp_writer.release()
            self.bounding_boxes = {}
            self.bounding_boxes[1] = self.load_bounding_boxes(f"{directory}/{filename}.txt", width, height)
            self.cap = cv2.VideoCapture(temp_path)
        else:
            # Check video format (if not I-frame, it should be re-encoded for proper use)
            video_format = check_video_format(video_path = file_path, time = 0.2)
            print(f"video_format: {video_format}")
            if video_format != "I":
                selected = self.playlist.currentRow()
                self.filelist.pop(selected)
                self.playlist.takeItem(selected)
                file_removed = True

                reply = QMessageBox.warning(self, "Warning", 
                    "This video is not I-frame format. Do you want to encode this video with I-frame format?",
                    QMessageBox.Yes | QMessageBox.No, 
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
                else:
                    self.converter_window = VideoConverterWindow(file_path)
                    self.converter_window.conversion_complete_signal.connect(self.on_conversion_complete)
                    self.converter_window.show()
            else:
                self.bounding_boxes = {}
                if os.path.exists(f"{directory}/{filename}.json"):
                    print(f"json file exist!")
                    with open(f"{directory}/{filename}.json", 'r') as f:
                        bboxes_str_keys = json.load(f)
                    # Convert keys back to integers and inner lists to tuples
                    self.bounding_boxes = {int(key): [tuple(bbox) for bbox in value] for key, value in bboxes_str_keys.items()}
                    
                self.cap = cv2.VideoCapture(file_path)
                if not self.cap:
                    print("Video is not loaded!")
                    return

        if file_removed == False:            
            self.current_filename = filename
            self.current_ext = ext
            self.current_directory = directory
            self.current_videopath = file_path
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
            # Initialization
            self.current_frame_n = 0
            self.playback_slider.setRange(1, self.total_frames)
            self.predicted_frame = [0]*(self.total_frames + 1)
            self.update_frame_label()
    
            # Zoom scale setting
            self.original_scale = min(self.video_player_width / self.video_width,
                                      self.video_player_height / self.video_height)
            self.zoom_scale = self.original_scale
            
            self.update_frame()
    ### File management end #######################
    
    ### Video controller start ####################
    def play_button_click(self):
        if self.cap:
            if self.is_playing == True:
                self.pause_video()
            else:
                self.play_video()
                
    def play_video(self):
        if self.cap:
            self.is_playing = True
            self.play_button.setText("⏸(Space)")
            self.video_timer.start(20)

    def pause_video(self):
        if self.cap:
            self.is_playing = False
            self.play_button.setText("▶(Space)")
            self.video_timer.stop()
    
    # ======== video slider controller ========
    def slider_pressed(self):
        if self.cap:
            self.pause_video()

    def slider_released(self):
        if self.cap:
            self.current_frame_n = int(self.playback_slider.value())
            self.refresh_frame()
    ### Video controller end ####################
    
    ### Load configuration start ################
    def load_config(self):
        try:
            with open("./config/AVATAR3D_config.json", "r") as f:
                config = json.load(f)
                self.classes = config.get("class", [])
                self.class_selector.clear()  # Initialize
                self.class_selector.addItems(self.classes)

                # 클래스별 고유 색상 할당
                self.class_colors = self.generate_class_colors(len(self.classes))
        except Exception as e:
            print(f"[Error] Failed to load classes: {e}")
    ### Load configuration end ################

    ### Yolo class start ######################
    def class_selected(self, index):
        self.current_class_id = index

    def generate_class_colors(self, num_classes):
        random.seed(42)  # 재현 가능한 색상 생성
        colors = []
        for _ in range(num_classes):
            colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        return colors
    ### Yolo class end ######################

    ### Yolo running start #################
    def save_yolo_result(self, results):
        self.bounding_boxes = results["bounding_boxes"]
        self.predicted_frame = results["predicted_frame"]

    def finish_yolo(self):
        self.statusBar().showMessage("Pose estimation completed!")
        self.refresh_frame()
        
    def run_yolo(self):
        if self.cap:            
            if self.cap_yolo:
                self.cap_yolo.release()
            
            self.cap_yolo = cv2.VideoCapture(self.current_videopath)
            self.statusBar().showMessage("Running yolo11..")
            self.yolo_task = YoloRunner(
                self.cap_yolo,
                self.predicted_frame,
                self.bounding_boxes
            )
            self.yolo_task.progress.connect(lambda msg: self.statusBar().showMessage(msg))
            self.yolo_task.result.connect(self.save_yolo_result)  # Result signal
            self.yolo_task.finished.connect(self.finish_yolo)
            self.yolo_task.stopped.connect(self.handle_task_stopped)
            self.yolo_task.start()
        else:
            QMessageBox.warning(self, "QMessageBox", "Please load video file")
    ### Yolo running end #################
    
    ### Update frame start #####################
    def render_frame(self, frame, selected = None):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ongoing Bounding Box
        if self.start_point and self.is_drawing == True:
            start_x = int((self.start_point[0] + self.h_scroll) / self.zoom_scale)
            start_y = int((self.start_point[1] + self.v_scroll) / self.zoom_scale)
            end_x = int((self.end_point[0] + self.h_scroll) / self.zoom_scale)
            end_y = int((self.end_point[1] + self.v_scroll) / self.zoom_scale)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

        # Bounding Box in current frame
        if self.current_frame_n in self.bounding_boxes:
            for bbox in self.bounding_boxes[self.current_frame_n]:
                class_id, x1, y1, x2, y2 = bbox
                color = self.class_colors[class_id % len(self.class_colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Highlight clicked Bbox
        if selected != None:
            class_id, x1, y1, x2, y2 = self.bounding_boxes[self.current_frame_n][selected]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
        q_image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))
        self.video_label.resize(int(self.video_width * self.zoom_scale), int(self.video_height * self.zoom_scale))
        self.update_frame_label()
        self.update_bbox_list()
        self.playback_slider.setValue(self.current_frame_n)
        
    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_n += 1
                self.current_frame = frame
                self.render_frame(self.current_frame)
            else:
                self.pause_video()
    
    def refresh_frame(self):
        if self.cap:
            self.current_frame_n = int(max(self.current_frame_n - 1, 0))
            if(self.current_frame_n == 0):
                self.cap.release()
                self.cap = cv2.VideoCapture(self.current_videopath)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_n)
            
            self.update_frame()
    
    def update_frame_label(self):
        self.frame_label.setText(f"Frame: {self.current_frame_n}/{self.total_frames}")
    ### Update frame end #####################

    ### Mouse control start #######################
    def mousePressEvent(self, event):
        if self.cap:
            self.pause_video()
            self.get_scroll_position()
            if (0 <= event.x() - 10 <= self.video_player_width and
                0 <= event.y() - 10 - self.menuBar().height() <= self.video_player_height):
                    self.start_point = (event.x() - 10, event.y() - 10 - self.menuBar().height())
                    if event.button() == Qt.LeftButton:    
                        self.is_drawing = True
                    elif event.button() == Qt.RightButton:
                        self.bbox_right_click_for_remove(self.start_point)
            
    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.pause_video()
            self.end_point = (min(event.x() - 10, self.video_player_width),
                              min(event.y() - 10 - self.menuBar().height(), self.video_player_height))
            
            self.render_frame(self.current_frame)

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
            temp_bbox = [self.current_class_id, *x1y1, *x2y2]
            
            if self.current_frame_n not in self.bounding_boxes:
                self.bounding_boxes[self.current_frame_n] = []
            self.bounding_boxes[self.current_frame_n].append(temp_bbox)

            self.update_bbox_list()

            self.render_frame(self.current_frame)
            
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
            self.render_frame(self.current_frame)
        else:
            super().wheelEvent(event)

    def get_scroll_position(self):
        self.h_scroll = self.scroll_area.horizontalScrollBar().value()
        self.v_scroll = self.scroll_area.verticalScrollBar().value()
    ### Mouse control end #######################

    ### Keyboard control start #######################
    def keyPressEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Ctrl 키와 함께 방향키 입력 시 프레임 이동
            if event.key() == Qt.Key_Right and self.is_playing == False:
                self.move_to_next_frame()
            elif event.key() == Qt.Key_Left and self.is_playing == False:
                self.move_to_previous_frame()
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
                
    def move_to_previous_frame(self):
        if self.cap:
            self.current_frame_n -= 1
            self.refresh_frame()
    
    def move_to_next_frame(self):
        if self.cap and self.current_frame_n < self.total_frames:
            self.update_frame()
    ### Keyboard control end #######################
    
    #### BBox management start #####################
    def update_bbox_list(self):
        self.bbox_list.clear()
        if self.current_frame_n in self.bounding_boxes:
            for bbox in self.bounding_boxes[self.current_frame_n]:
                class_id, x1, y1, x2, y2 = bbox
                self.bbox_list.addItem(
                    f"Class: {self.classes[class_id]} | ({x1}, {y1}) -> ({x2}, {y2})"
                )
    
    def save_files(self, mode):
        if self.current_ext == ".mp4":
            print(self.bounding_boxes)
            save_path = f"{self.current_directory}/{self.current_filename}.json"
            with open(save_path, 'w') as f:
                json.dump(self.bounding_boxes, f, indent=4)
        
        if self.cap:
            self.cap_save = cv2.VideoCapture(self.current_videopath)
            self.statusBar().showMessage("Saving..")
            self.save_task = YoloSaver(
                self.cap_save,
                self.predicted_frame,
                self.bounding_boxes,
                self.current_frame_n,
                self.current_videopath,
                mode
            )
            self.save_task.progress.connect(lambda msg: self.statusBar().showMessage(msg))
            self.save_task.finished.connect(lambda: self.statusBar().showMessage("Save completed!"))
            self.save_task.stopped.connect(self.handle_task_stopped)
            self.save_task.start()
        
    def delete_selected_box(self):
        selected = self.bbox_list.currentRow()
        if selected >= 0:
            self.bounding_boxes[self.current_frame_n].pop(selected)
            self.bbox_list.takeItem(selected)
            self.render_frame(self.current_frame)
        
    def selected_bbox(self, item):
        row = self.bbox_list.row(item)
        self.render_frame(self.current_frame, selected = row)
        # item.setSelected(True)

    def bbox_right_click_for_remove(self, xy):
        click_x = int((xy[0] + self.h_scroll) / self.zoom_scale)
        click_y = int((xy[1] + self.v_scroll) / self.zoom_scale)
        selected = None
        closest_distance = float("inf")
        threshold = 10  # Distance threshold in pixels for proximity click
            
        # Find the closest bbx by minimum edge distance
        idx = 0
        for bbox in self.bounding_boxes[self.current_frame_n]:
            _, x1, y1, x2, y2 = bbox
            min_distance = calc_min_distance(click_x, click_y, x1, y1, x2, y2)
    
            # Check if this rectangle is the closest one within the threshold
            if min_distance < closest_distance and min_distance <= threshold:
                closest_distance = min_distance
                selected = idx
            idx += 1

        # If a rectangle is close enough, delete it
        if selected is not None:
            self.bounding_boxes[self.current_frame_n].pop(selected)
            self.bbox_list.takeItem(selected)
            self.render_frame(self.current_frame)
    #### BBox management end #####################

    ### Task management start #####################
    def stop_save_task(self):
        if hasattr(self, 'yolo_task') and self.yolo_task.isRunning():
            self.yolo_task.stop()
            self.cap_yolo.release()
        if hasattr(self, 'save_task') and self.save_task.isRunning():
            self.save_task.stop()
            self.cap_save.release()

    def handle_task_stopped(self):
        self.statusBar().showMessage("Task is stopped!")
    ### Task management end #####################

    ### Video conversion start #####################
    def on_conversion_complete(self, video_path):
        self.filelist.append(video_path)
        self.playlist.addItems([video_path])
        self.statusBar().showMessage("Video conversion completed and added in the filelist")
    ### Video conversion end #####################

    def closeEvent(self, event):
        if self.converter_window is not None:
            self.converter_window.close()
            event.accept()
        else:
            event.accept()

            
class VideoConverterWindow(QWidget):
    conversion_complete_signal = pyqtSignal(str)
    
    def __init__(self, input_file):
        super().__init__()
        self.setWindowTitle("Video Converter")
        self.setGeometry(100, 100, 200, 100)

        self.input_file = input_file
        
        self.init_gui()

    def init_gui(self):
        layout = QVBoxLayout()

        # Progress Bar
        self.progress_label = QLabel("Progress: 0.00%")
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)

        # Start and Stop Buttons
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Conversion")
        self.start_button.clicked.connect(self.start_conversion)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Conversion")
        self.stop_button.clicked.connect(self.stop_conversion)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def start_conversion(self):
        self.worker = ConvertVideoToIframe(self.input_file)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.status_signal.connect(self.update_status)
        self.worker.result.connect(self.on_conversion_finished)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.worker.start()

    def stop_conversion(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
            self.close()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)
        self.progress_label.setText(f"Progress: {progress:.2f}%")

    def update_status(self, status):
        self.progress_label.setText(status)

    def on_conversion_finished(self, video_path):
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.conversion_complete_signal.emit(video_path)
        self.close()
        
### Main program start ###############
if __name__ == "__main__":
    # multiprocessing.freeze_support()
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    player = IBS_BAP()
    player.show()
    player.raise_()
    player.activateWindow()

    try:
        app.exec_()
    except SystemExit:
        print("[Info] PyQt5 Application exited cleanly.")
    finally:
        app.quit()
        del app
        print("[Info] QApplication resources have been cleaned up.")
### Main program end ###############