import cv2
import numpy as np
import os
import uuid

# COCO 關鍵點定義 (17 個點的順序)
# 確保這個列表與您的 API 返回的關鍵點順序一致
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle"
]
# COCO 骨架連接關係 (基於 COCO_KEYPOINTS 的索引)
COCO_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 頭部和臉: 鼻子-左眼, 鼻子-右眼, 左眼-左耳, 右耳-右耳
    (5, 6),  # 軀幹: 左肩-右肩
    (5, 7), (7, 9),  # 左臂: 左肩-左肘, 左肘-左手腕
    (6, 8), (8, 10), # 右臂: 右肩-右肘, 右肘-右手腕
    (5, 11), (6, 12), (11, 12), # 軀幹和臀部: 左肩-左髖, 右肩-右髖, 左髖-右髖
    (11, 13), (13, 15), # 左腿: 左髖-左膝, 左膝-左腳踝
    (12, 14), (14, 16)  # 右腿: 右髖-右膝, 右膝-右腳踝
]

def draw_pose_on_frame(frame, predictions, min_score_thresh=0.5, point_radius=5, line_thickness=2):
    """
    在單個影片幀上繪製姿態關鍵點和骨架。

    Args:
        frame (np.array): OpenCV 圖片幀 (BGR 格式)。
        predictions (list): API 回應中針對該幀的 'predictions' 列表。
                            預期格式: [{'keypoints': [[x, y], ...], 'keypoint_scores': [...], ...}]
        min_score_thresh (float): 關鍵點顯示的最低置信度閾值。
        point_radius (int): 關鍵點圓圈的半徑。
        line_thickness (int): 骨架線條的粗細。

    Returns:
        np.array: 繪製了姿態的幀 (BGR 格式)。
    """
    if not predictions:
        return frame # 如果沒有偵測到姿態，直接返回原幀

    # 針對每個偵測到的人繪製姿態
    for person_prediction in predictions:
        keypoints_coords = person_prediction.get('keypoints', [])
        keypoint_scores = person_prediction.get('keypoint_scores', [])

        if not keypoints_coords or not keypoint_scores:
            continue

        # 將關鍵點座標和分數打包成列表，方便後續處理
        # 這裡的 keypoints_data 會是 [(x, y, score), (x, y, score), ...]
        keypoints_data = []
        for i, (x, y) in enumerate(keypoints_coords):
            score = keypoint_scores[i] if i < len(keypoint_scores) else 0.0
            keypoints_data.append((int(x), int(y), score)) # 轉換為整數像素座標

        # 繪製關鍵點
        for kp_idx, (x, y, score) in enumerate(keypoints_data):
            if score > min_score_thresh:
                # 關鍵點顏色 (B, G, R) - 例如，綠色
                cv2.circle(frame, (x, y), point_radius, (0, 255, 0), -1)

        # 繪製骨架連接
        for connection in COCO_CONNECTIONS:
            start_kp_idx = connection[0]
            end_kp_idx = connection[1]

            if start_kp_idx < len(keypoints_data) and end_kp_idx < len(keypoints_data):
                x1, y1, s1 = keypoints_data[start_kp_idx]
                x2, y2, s2 = keypoints_data[end_kp_idx]

                if s1 > min_score_thresh and s2 > min_score_thresh:
                    # 連線顏色 (B, G, R) - 例如，黃色
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), line_thickness)
    return frame

def render_video_with_pose(video_path: str, api_response_json: dict, output_dir: str) -> str:
    """
    將姿態偵測結果渲染到原始影片的每一幀上，並將其保存為新的影片檔案。

    Args:
        video_path (str): 原始影片文件的路徑。
        api_response_json (dict): 從 API 獲得的完整 JSON 回應。
        output_dir (str): 處理後影片保存的目錄。

    Returns:
        str: 已保存的處理過影片檔案的路徑。如果處理失敗，則返回空字符串。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 為處理過的影片生成一個獨特的檔名
    processed_video_id = str(uuid.uuid4())[:8]
    processed_video_filename = f"{processed_video_id}_pose_rendered.mp4"
    processed_video_path = os.path.join(output_dir, processed_video_filename)

    # 定義編解碼器並創建 VideoWriter 物件
    # 對於 .mp4 輸出，使用 'mp4v'。如果遇到問題，可以嘗試 'avc1' 或確保 FFmpeg 可用。
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # For .mp4
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not open video writer for {processed_video_path}. Check codec or path permissions.")
        cap.release()
        return ""

    api_frames_data = {frame_data['frame_idx']: frame_data for frame_data in api_response_json.get('frames', [])}

    current_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # 影片讀取結束

        # 從 API 回應中獲取當前幀的姿態數據
        predictions_raw = api_frames_data.get(current_frame_idx, {}).get('predictions', [])

        # 根據用戶提供的數據結構修正：predictions 是一個包含一個列表的列表
        # 例如: [[{'keypoints': ...}, {'keypoints': ...}]]
        # 我們需要取出內層的列表
        predictions_for_current_frame = []
        if predictions_raw and isinstance(predictions_raw, list) and predictions_raw[0] and isinstance(predictions_raw[0], list):
            predictions_for_current_frame = predictions_raw[0]
        elif predictions_raw and isinstance(predictions_raw, list) and predictions_raw[0] and isinstance(predictions_raw[0], dict):
            # 如果未來 API 結構改變，直接就是字典列表，這裡也可以兼容
            predictions_for_current_frame = predictions_raw

        # 在幀上繪製姿態
        rendered_frame = draw_pose_on_frame(
            frame.copy(), # 傳遞副本以避免修改原始幀
            predictions_for_current_frame
        )

        out.write(rendered_frame) # 寫入處理過的幀
        current_frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished rendering {current_frame_idx} frames and saved to {processed_video_path}.")
    return processed_video_path
