from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid, os, shutil, json
import requests

# 從新的檔案導入影片處理函數
from pose_renderer import render_video_with_pose # 確保 pose_renderer.py 在同一個目錄下

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/videos"
PROCESSED_DIR = "static/processed_videos" # Directory for processed videos
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# 模擬姿勢資料
def mock_posture(video_id: str):
    return {
        "stride_angle": 42.5,
        "throwing_angle": 95.3,
        "arm_symmetry": 88.0,
        "hip_rotation": 35.2,
        "elbow_height": 123
    }

# 模擬模型預測
def mock_prediction(video_id: str):
    return {
        "result": "Good" if video_id.endswith("1") else "Bad",
        "confidence": 0.92 if video_id.endswith("1") else 0.65
    }

# 載入歷史
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"history": []}

# 寫入歷史
def append_history(filename, result):
    history = load_history()
    history["history"].append({
        "timestamp": "2025-06-22",
        "filename": filename,
        "result": result
    })
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


POSE_API_URL = "https://mmpose-api-924124779607.us-central1.run.app/pose_video"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())[:8]
    original_save_path = os.path.join(UPLOAD_DIR, f"{video_id}{extension}")

    # 保存原始上傳影片
    with open(original_save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pose_api_response = {}
    try:
        # 以二進制模式讀取已保存的影片檔案以進行 POST 請求
        with open(original_save_path, "rb") as video_file_data:
            files = {'video': (file.filename, video_file_data, 'video/mp4')}
            response = requests.post(POSE_API_URL, files=files, timeout=300) # 5 分鐘超時
            response.raise_for_status()
            pose_api_response = response.json()
            print(f"Pose API Response: {pose_api_response}")
    except requests.exceptions.RequestException as e:
        print(f"Error calling pose API: {e}")
        pose_api_response = {"error": True, "message": f"Failed to get pose data from external API: {e}"}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from pose API: {e}")
        pose_api_response = {"error": True, "message": f"Invalid JSON response from external API: {e}"}


    # 渲染帶有姿勢資料的影片，現在從 pose_renderer 導入
    processed_video_local_path = render_video_with_pose(original_save_path, pose_api_response, PROCESSED_DIR)

    if not processed_video_local_path:
        # 如果處理失敗，則回退到原始影片
        processed_video_url = f"/static/videos/{video_id}{extension}"
        print("Falling back to original video URL due to processing failure.")
    else:
        # 構建處理過影片的 URL
        processed_video_url = f"/static/processed_videos/{os.path.basename(processed_video_local_path)}"

    # 模擬預測並寫入歷史記錄 (保留用於儀表板其他部分的現有邏輯)
    result = mock_prediction(video_id)["result"]
    append_history(f"{video_id}{extension}", result)

    return {
        "video_id": video_id,
        "original_video_url": f"/static/videos/{video_id}{extension}",
        "processed_video_url": processed_video_url,
        "pose_data": pose_api_response
    }

@app.get("/analysis/{video_id}")
async def get_analysis(video_id: str):
    return mock_posture(video_id)

@app.get("/predict/{video_id}")
async def get_prediction(video_id: str):
    return mock_prediction(video_id)

@app.get("/history")
async def get_history():
    return load_history()

# 靜態檔案服務
app.mount("/static", StaticFiles(directory="static"), name="static")
