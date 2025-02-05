import cv2
import numpy as np
import json
import os

from matplotlib import pyplot as plt
from PyQt5.QtCore import QTimer, Qt, QPoint, QThread, pyqtSignal

class Recon3D(QThread):
    finished = pyqtSignal()
    
    def __init__(self, input_path, raw_coordinates):
        super().__init__()
        self.input_path = input_path
        self.raw_coordinates = raw_coordinates
        
        directory, filename = os.path.split(self.input_path)
        filename, ext = os.path.splitext(filename)
        self.output_path = f"{directory}/{filename}_coordinates.csv"

        # Load configuration file
        config_file = './config/AVATAR3D_config.json'
        with open(config_file, 'r') as file:
            self.cfg = json.load(file)
        
        self.camera_matrices = [np.array(mtx) for mtx in self.cfg["camera"]["calibration"]["camera_matrices"]]
        self.dist_coeffs = [np.array(dist) for dist in self.cfg["camera"]["calibration"]["dist_coeffs"]]
        self.rot_vectors = [np.array(rvec) for rvec in self.cfg["camera"]["calibration"]["rot_vectors"]]
        self.trans_vectors = [np.array(tvec) for tvec in self.cfg["camera"]["calibration"]["trans_vectors"]]
        self.square_size = self.cfg["camera"]["calibration"]["square_size_overide_mm"]
        
        # 각 카메라의 투영 행렬 계산
        self.projection_matrices = []
        for cam_idx, (K, rvec, tvec) in enumerate(zip(self.camera_matrices, self.rot_vectors, self.trans_vectors)):
            R, _ = cv2.Rodrigues(rvec)  # 회전 벡터를 회전 행렬로 변환
            T = tvec.reshape(3, 1)          # 변환 벡터
            RT = np.hstack((R, T))          # [R | T]
            P = np.dot(K, RT)               # 투영 행렬: K * [R | T]
            self.projection_matrices.append(P)
        
        # 평행이동과 회전을 위한 value: pre-determined
        self.INTERSECTION = [ 5.485472, 11.23394,  -6.582967]
        self.ROTATION_MATRIX = [[ 0.9940044, -0.09772341, -0.04904471],
         [-0.09772341, -0.59281272, -0.79938928],
         [0.04904471, 0.79938928, -0.59880832]]
        self.INTERSECTION = np.array(self.INTERSECTION)
        self.ROTATION_MATRIX = np.array(self.ROTATION_MATRIX)

    def transform_point(self, point):
        translated_point = point - self.INTERSECTION
        aligned_point = np.dot(translated_point, self.ROTATION_MATRIX.T)
        return aligned_point
    
    def triangulate_points(self, row, conf, camera_cnt):
        points_2d = np.array(row).reshape(5, 2)  # (x, y) 좌표로 변환
        
        # 왜곡 보정된 좌표로 변환
        undistorted_points = []
        for cam_idx, (pt, K, dist) in enumerate(zip(points_2d, self.camera_matrices, self.dist_coeffs)):
            pts = np.array([pt], dtype=np.float32).reshape(-1, 1, 2)
            undistorted = cv2.undistortPoints(pts, K, dist, None, K).reshape(-1, 2)
            undistorted_points.append(undistorted[0])
    
        # conf 높은 순으로 index 정렬
        sorted_indices = sorted(range(len(conf)), key=lambda i: conf[i], reverse=True)
        
        # 최소 두 카메라를 선택하여 삼각화 수행: (0,0)이 아닌 카메라 2개 찾기
        cam_idx = []
        points = []
        cnt = 0
        for i in sorted_indices:
            if(conf[i] > 0.25):
                cam_idx.append(i)
                points.append(np.array(undistorted_points[i], dtype=np.float32).reshape(2, 1))
                cnt += 1
    
        if(cnt < 2):
            return [0.0, 0.0, 0.0]
        else:   
            # Triangulation: 2D 좌표에서 3D 좌표 계산
            triangulated_point = cv2.triangulatePoints(
                self.projection_matrices[cam_idx[0]], self.projection_matrices[cam_idx[1]],
                points[0], points[1]
            )
        
            # 동차 좌표를 디카르트 좌표로 변환
            triangulated_point /= triangulated_point[3]
    
            # square_size 를 곱해서 실제 위치 계산
            triangulated_point = triangulated_point * self.square_size
    
            return triangulated_point[:3].flatten()

    def run(self):
        # Load points from file and process
        # points_data = np.loadtxt(input_file, delimiter="\t")  # Load the data from file
        points_data = self.raw_coordinates
        
        # Extract 2D points and confidences
        triangulated_points = []
        for row in points_data:
            triangulated_row = []  # Store triangulated points for one row
            for i in range(self.cfg['keypoint']['cnt']):
                points_2d = []
                conf = []
                for j in range(self.cfg['camera']['cnt']):
                    x, y, confidence = row[j*self.cfg['keypoint']['cnt']*3 + i*3 : j*self.cfg['keypoint']['cnt']*3 + i*3 + 3]
                    points_2d.append(x - self.cfg['camera']['roi'][j][0])
                    points_2d.append(y - self.cfg['camera']['roi'][j][1])
                    conf.append(confidence)
                
                point_3d = self.triangulate_points(points_2d, conf, self.cfg['camera']['cnt'])
                transformed_point = self.transform_point(point_3d) # 카메라 3을 z축 기준으로 전환
                
                triangulated_row.extend(transformed_point)
            
            triangulated_points.append(triangulated_row)
        
        # Save results to file
        np.savetxt(self.output_path, triangulated_points, fmt="%.6f", comments="", delimiter=",")