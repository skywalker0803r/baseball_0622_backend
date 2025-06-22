from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid, os, shutil, json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可調整為前端網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "static/videos"
HISTORY_FILE = "history.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

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

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    extension = os.path.splitext(file.filename)[1]
    video_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}{extension}")

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    video_url = f"/static/videos/{video_id}{extension}"
    result = mock_prediction(video_id)["result"]
    append_history(f"{video_id}{extension}", result)

    return {
        "video_id": video_id,
        "video_url": video_url
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

# 靜態檔案服務 (搭配 uvicorn 建議另用 nginx)
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")
