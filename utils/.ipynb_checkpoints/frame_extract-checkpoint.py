import cv2
import os
import multiprocessing
import time

def save_frame(frame_data):
    frame, count, output_folder = frame_data
    cv2.imwrite(os.path.join(output_folder, f"frame{count:06d}.jpg"), frame)

def save_frames_multiprocessing(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
      print("비디오 파일을 열 수 없습니다.")
      return
        
    success, image = vidcap.read()
    count = 0
    frame_data_list = []

    start_read_time = time.time() # 프레임 읽기 시작 시간
    while success:
        frame_data_list.append((image, count, output_folder))
        success, image = vidcap.read()
        count += 1
    end_read_time = time.time() # 프레임 읽기 완료 시간
    
    vidcap.release()

    start_save_time = time.time() # 프레임 저장 시작 시간
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(save_frame, frame_data_list)
    end_save_time = time.time() # 프레임 저장 완료 시간

    total_time = end_save_time - start_save_time + end_read_time - start_read_time
    read_time = end_read_time - start_read_time
    save_time = end_save_time - start_save_time

    print(f"총 {count}개의 프레임")
    print(f"프레임 읽기 시간: {read_time:.2f} 초")
    print(f"프레임 저장 시간: {save_time:.2f} 초")
    print(f"총 걸린 시간: {total_time:.2f} 초")

if __name__ == "__main__":
    video_path = "D:/spkim/data/20241114_40Hz_test/20241114_40Hz_test/C_20Hz_Bandi_20fps.mp4"
    output_folder = "output_frames"
    save_frames_multiprocessing(video_path, output_folder)
    print("프레임 저장 완료")
