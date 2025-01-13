# 2024-01-13 ver
import sys
import cv2
import numpy as np
import json
import random
import os
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QSlider, QHBoxLayout, QListWidget, QListWidgetItem,
    QComboBox, QScrollArea, QMenuBar, QAction, QMessageBox, QDialog
)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

class LibraryLoaderThread(QThread):
    # 라이브러리 로드 완료 신호를 위한 시그널
    finished = pyqtSignal(str)

    def run(self):
        try:
            # 프로젝트 루트 경로 추가
            project_root = Path.cwd()  # Jupyter Notebook에서는 cwd를 사용
            sys.path.append(str(project_root / "utils"))
                        
            result = "Library 'yolo' loaded successfully!"
            print("libraries are loaded!")
        except ImportError as e:
            result = f"Failed to load library: {e}"
        
        # 결과를 메인 스레드로 전달
        self.finished.emit(result)

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IBS Behavior Analysis Program (BAP)")
        self.setGeometry(100, 100, 1200, 700)
        #### Menu bar ########################
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_file_action = QAction("Open Files", self)
        open_file_action.triggered.connect(self.open_files)
        file_menu.addAction(open_file_action)
        
        open_action = QAction("Open Video", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
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
        
        # ======== UI 구성 ========
        self.video_player_width = 1300
        self.video_player_height = 800
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
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setFixedWidth(120)
        self.frame_cache = {}

        # 드롭다운 (Class 선택)
        self.class_selector = QComboBox()
        
        # 버튼
        self.play_button = QPushButton("▶(Space)") # "⏸ Pause"
        self.previous_frame_button = QPushButton("< (d)")
        self.next_frame_button = QPushButton("> (f)")
        self.save_box_button = QPushButton("💾 Save current BBox")
        self.save_all_button = QPushButton("💾 Save All BBox")
        self.delete_box_button = QPushButton("❌ Delete Selected Box")
        self.yolo_button = QPushButton("Run yolo11")
        self.stop_button = QPushButton("🛑 Stop Task")

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
        self.video_files = []
        self.total_frames = 0
        self.fps = 0
        self.current_frame = None
        self.current_frame_n = 0
        self.current_filename = None
        self.current_ext= None
        self.current_videopath = None
        
        # ==== Timer setting ========
        self.video_timer = QTimer(self)
        self.video_timer.timeout.connect(self.update_frame)

        # ======== Bounding Box ========
        self.start_point = None
        self.end_point = None
        self.is_drawing = False
        self.bounding_boxes = {}

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
        self.predicted_frame = []

        # ======== 이벤트 핸들러 연결 ========
        self.play_button.clicked.connect(self.play_button_click)
        self.previous_frame_button.clicked.connect(self.move_to_previous_frame)
        self.next_frame_button.clicked.connect(self.move_to_next_frame)
        self.save_box_button.clicked.connect(lambda: self.save_files("one"))
        self.save_all_button.clicked.connect(lambda: self.save_files("all"))
        self.delete_box_button.clicked.connect(self.delete_selected_box)
        self.playlist.itemClicked.connect(self.play_selected_video)
        self.playback_slider.sliderPressed.connect(self.slider_pressed)
        self.playback_slider.sliderReleased.connect(self.slider_released)
        self.class_selector.currentIndexChanged.connect(self.class_selected)
        self.yolo_button.clicked.connect(self.run_yolo)
        self.stop_button.clicked.connect(self.stop_save_task)

        # Thread for importing YOLO libraries (ultralytics)
        self.model = {}
        self.library_thread = LibraryLoaderThread()
        self.library_thread.finished.connect(self.on_library_loaded)
        self.library_thread.start()

    def on_library_loaded(self, message):
        self.yolo_button.setText("Run yolo11")
        self.yolo_button.setEnabled(True)
    
    # ======== 동영상 열기 ========
    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Files", "", "Video Files (*.mp4 *.jpg *.png)")
        if files:
            self.video_files.extend(files)
            self.playlist.addItems(files)
    
    def open_video(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Video Files", "", "Video Files (*.mp4 *.avi *.mkv)")
        if files:
            self.video_files.extend(files)
            self.playlist.addItems(files)

    def open_image_folder(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Open Image Folders", "", "Image Files (*.jpg *.png)")
        if files:
            self.in_files.extend(files)
            self.playlist.addItems(files)
        
    def play_selected_video(self, item: QListWidgetItem):
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
        self.current_filename, self.current_ext = os.path.splitext(file_path.split("/")[-1])
        self.current_videopath = file_path
        
        if self.current_ext in [".jpg", ".png"]: # Treat as one frame video
            self.current_frame = cv2.imread(file_path)
            height, width, _ = self.current_frame.shape
            file_path = "temp.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_writer = cv2.VideoWriter(file_path, fourcc, 1, (width, height))
            temp_writer.write(self.current_frame)
            temp_writer.release()

        # Capture video and get various values
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap:
            print("Video is not loaded!")
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Initialization
        self.current_frame_n = 0
        self.playback_slider.setRange(0, self.total_frames)
        self.predicted_frame = [0]*(self.total_frames + 1)
        self.update_frame_label()
        self.bounding_boxes = {}

        # Zoom scale setting
        self.original_scale = min(self.video_player_width / self.video_width,
                                  self.video_player_height / self.video_height)
        self.zoom_scale = self.original_scale
        
        self.update_frame()

    # ======== Video controller ============
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

    def save_yolo_result(self, results):
        self.bounding_boxes = results["bounding_boxes"]
        self.predicted_frame = results["predicted_frame"]

    def finish_yolo(self):
        self.statusBar().showMessage("Pose estimation completed!")
        self.refresh_frame()
        
    def run_yolo(self):
        if self.cap:
            self.statusBar().showMessage("Load YOLO model..")
            from utils import YoloRunner
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
            QMessageBox.warning(self, "QMessageBox", "Please load the video")

    # ======== Update frame ========
    def render_frame(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ongoing Bounding Box
        if self.start_point and self.is_drawing == True:
            start_x = int((self.start_point[0] + self.h_scroll) / self.zoom_scale)
            start_y = int((self.start_point[1] + self.v_scroll) / self.zoom_scale)
            end_x = int((self.end_point[0] + self.h_scroll) / self.zoom_scale)
            end_y = int((self.end_point[1] + self.v_scroll) / self.zoom_scale)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3)

        # Bounding Box in current frame
        if self.current_frame_n in self.bounding_boxes:
            for bbox in self.bounding_boxes[self.current_frame_n]:
                class_id, x1, y1, x2, y2 = bbox
                color = self.class_colors[class_id % len(self.class_colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
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
                self.current_frame = frame
                self.render_frame(self.current_frame)
                # if len(self.frame_cache) > 50:
                #     self.frame_cache.pop(next(iter(self.frame_cache)))
                self.current_frame_n += 1
            else:
                self.pause_video()

    # ======== video slider controller ========
    def slider_pressed(self):
        if self.cap:
            self.pause_video()

    def slider_released(self):
        if self.cap:
            self.current_frame_n = int(self.playback_slider.value())
            print(self.current_frame_n)
            self.refresh_frame()

    # ======== Bounding Box 이벤트 ========
    def mousePressEvent(self, event):
        if self.cap:
            self.get_scroll_position()
            if (event.button() == Qt.LeftButton and
                0 <= event.x() - 10 <= self.video_player_width and
                0 <= event.y() - 10 - self.menuBar().height() <= self.video_player_height):
                
                self.pause_video()
                self.start_point = (event.x() - 10, event.y() - 10 - self.menuBar().height())
                self.is_drawing = True

    def mouseMoveEvent(self, event):
        if self.cap and self.is_drawing:
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
            temp_bbox = (self.current_class_id, *x1y1, *x2y2)
            
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
        
    # ======== Bounding Box save or delete ========
    def update_bbox_list(self):
        self.bbox_list.clear()
        if self.current_frame_n in self.bounding_boxes:
            for bbox in self.bounding_boxes[self.current_frame_n]:
                class_id, x1, y1, x2, y2 = bbox
                self.bbox_list.addItem(
                    f"Class: {self.classes[class_id]} | ({x1}, {y1}) -> ({x2}, {y2})"
                )

    def save_files(self, mode):
        if self.cap:
            from utils import YoloSaver
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

    def move_to_next_frame(self):
        if self.cap and self.current_frame_n < self.total_frames:
            self.update_frame()

    def move_to_previous_frame(self):
        if self.cap:
            self.current_frame_n -= 1
            self.refresh_frame()
            
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
        self.frame_label.setText(f"Frame: {self.current_frame_n + 1}/{self.total_frames}")
    
    def reset_statusBar(self):
        self.status_label.setText("IBS_BAP by Sunpil Kim")

    def stop_save_task(self):
        if hasattr(self, 'yolo_task') and self.yolo_task.isRunning():
            self.yolo_task.stop()
            self.cap_yolo.release()
        if hasattr(self, 'save_task') and self.save_task.isRunning():
            self.save_task.stop()
            self.cap_save.release()

    def handle_task_stopped(self):
        self.statusBar().showMessage("Task is stopped!")
        
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