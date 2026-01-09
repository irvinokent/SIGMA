import os
import cv2
import numpy as np
import pandas as pd

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
try:
    from ultralytics import RTDETR
    HAS_RTDETR = True
except ImportError:
    RTDETR = None
    HAS_RTDETR = False

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models.detection import fcos_resnet50_fpn

# ====================== CONFIG GLOBAL ======================
# Path model
YOLO_MODEL_PATH   = "yolov11n_visdrone_5cls_bikemoto_ft.pt"
FCOS_MODEL_PATH   = "fcos.pth"
RTDETR_MODEL_PATH = "rtdetr_visdrone_5cls.pt"  # GANTI sesuai file RT-DETR-mu

CONF_THRESH = 0.35
IMGSZ = 640  # dipakai FCOS & RT-DETR; YOLO pakai 1280 di bawah

YOLO_CONF_THRESH = 0.15
YOLO_IMGSZ = 1280

RTDETR_CONF_THRESH = 0.15
RTDETR_IMGSZ = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VEHICLE_CLASSES = {"bicycle", "car", "truck", "bus", "motorcycle"}
PCU = {
    "motorcycle": 0.5,
    "car": 1.0,
    "bicycle": 0.4,
    "kendaraan_besar": 2.0,
}
MIN_GREEN_FUZZY = 15
MAX_GREEN_FUZZY = 45

CLASSES_FCOS_RT = ["bicycle", "car", "truck", "bus", "motorcycle"]
to_tensor = T.ToTensor()

# ====================== FASTAPI SETUP ======================
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("YOLO Loaded.")

print("Loading FCOS model...")
def load_fcos_model(model_path: str):
    num_classes = len(CLASSES_FCOS_RT) + 1  # +1 background
    model = fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=num_classes,
        min_size=IMGSZ,
        max_size=IMGSZ,
    )
    state = torch.load(model_path, map_location=DEVICE)
    if isinstance(state, dict):
        for k in ["model", "model_state", "state_dict"]:
            if k in state:
                state = state[k]
                break
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

fcos_model = load_fcos_model(FCOS_MODEL_PATH)
print("FCOS Loaded.")

if HAS_RTDETR:
    try:
        print("Loading RT-DETR model...")
        rtdetr_model = RTDETR(RTDETR_MODEL_PATH)
        print("RT-DETR Loaded.")
    except Exception as e:
        print(f"RT-DETR failed to load: {e}")
        rtdetr_model = None
else:
    rtdetr_model = None
    print("RT-DETR class not available in ultralytics. Skipping RT-DETR.")

# ====================== STATE UNTUK PICO ======================
LAST_FUZZY = {
    "UTARA":   {"Green_time": 10.0, "Red_time": 50.0},
    "TIMUR":   {"Green_time": 10.0, "Red_time": 50.0},
    "SELATAN": {"Green_time": 10.0, "Red_time": 50.0},
    "BARAT":   {"Green_time": 10.0, "Red_time": 50.0},
}

# ====================== HELPER: KATEGORI & OVERLAY ======================
def kategori_kendaraan(label: str):
    if label in ("truck", "bus"):
        return "kendaraan_besar"
    if label == "motorcycle":
        return "motorcycle"
    if label == "bicycle":
        return "bicycle"
    if label == "car":
        return "car"
    return None

def draw_overlay(frame, dets, agg_counts, agg_pcu):
    img = frame.copy()

    for d in dets:
        x1, y1, x2, y2 = d["box_xyxy"]
        label = d["label"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (10, 255, 20), 2)
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (10, 255, 20),
            2,
        )

    y0 = 30
    lines = [
        f"TOTAL PCU: {agg_pcu:.1f}",
        f"car: {agg_counts['car']}",
        f"motorcycle: {agg_counts['motorcycle']}",
        f"bicycle: {agg_counts['bicycle']}",
        f"kendaraan_besar: {agg_counts['kendaraan_besar']}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            img,
            line,
            (10, y0 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    return img

# ====================== HELPER: DETEKSI PER MODEL ======================
def detect_yolo(bgr):
    results = yolo_model(
        bgr,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF_THRESH,
        iou=0.6,
        max_det=500,
        verbose=False,
    )

    r = results[0]
    dets = []
    agg_counts = {
        "kendaraan_besar": 0,
        "car": 0,
        "motorcycle": 0,
        "bicycle": 0,
    }
    agg_pcu = 0.0

    if r is not None and r.boxes is not None and len(r.boxes) > 0:
        names = yolo_model.names
        for b in r.boxes:
            cls_name = names[int(b.cls.item())]
            if cls_name not in VEHICLE_CLASSES:
                continue

            kat = kategori_kendaraan(cls_name)
            if kat is None:
                continue

            agg_counts[kat] += 1
            agg_pcu += PCU.get(kat, 0.0)

            x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy().ravel())
            dets.append({"label": kat, "box_xyxy": [x1, y1, x2, y2]})

    return dets, agg_counts, agg_pcu

def detect_fcos(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    img_tensor = to_tensor(pil_img).to(DEVICE)

    with torch.no_grad():
        outputs = fcos_model([img_tensor])[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    dets = []
    agg_counts = {
        "kendaraan_besar": 0,
        "car": 0,
        "motorcycle": 0,
        "bicycle": 0,
    }
    agg_pcu = 0.0

    for box, lbl, score in zip(boxes, labels, scores):
        if score < CONF_THRESH:
            continue

        cls_id = int(lbl)
        if cls_id <= 0 or cls_id > len(CLASSES_FCOS_RT):
            continue

        cls_name = CLASSES_FCOS_RT[cls_id - 1]
        if cls_name not in VEHICLE_CLASSES:
            continue

        kat = kategori_kendaraan(cls_name)
        if kat is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        agg_counts[kat] += 1
        agg_pcu += PCU.get(kat, 0.0)

        dets.append(
            {
                "label": kat,
                "box_xyxy": [x1, y1, x2, y2],
            }
        )

    return dets, agg_counts, agg_pcu

def detect_rtdetr(bgr):
    """
    Deteksi kendaraan pakai RT-DETR (Ultralytics).
    Alur sama seperti script Streamlit:
    - infer RT-DETR
    - filter hanya kelas kendaraan
    - hitung PCU & jumlah per kategori
    """
    if rtdetr_model is None:
        # Kalau RT-DETR gagal load: kembalikan nol semua
        agg_counts = {
            "kendaraan_besar": 0,
            "car": 0,
            "motorcycle": 0,
            "bicycle": 0,
        }
        return [], agg_counts, 0.0

    results = rtdetr_model(
        bgr,
        imgsz=RTDETR_IMGSZ,        # ⬅️ pakai IMGSZ khusus RT-DETR
        conf=RTDETR_CONF_THRESH,   # ⬅️ pakai CONF THRESH khusus RT-DETR
        iou=0.6,
        max_det=500,
        verbose=False,
    )

    r = results[0]
    dets = []
    agg_counts = {
        "kendaraan_besar": 0,
        "car": 0,
        "motorcycle": 0,
        "bicycle": 0,
    }
    agg_pcu = 0.0

    if r is not None and r.boxes is not None and len(r.boxes) > 0:
        names = rtdetr_model.names  # RT-DETR juga punya .names seperti YOLO

        for b in r.boxes:
            cls_id = int(b.cls.item())
            cls_name = names[cls_id]

            # Hanya kelas kendaraan
            if cls_name not in VEHICLE_CLASSES:
                continue

            kat = kategori_kendaraan(cls_name)
            if kat is None:
                continue

            xyxy = b.xyxy.cpu().numpy().ravel()
            x1, y1, x2, y2 = map(int, xyxy)

            agg_counts[kat] += 1
            agg_pcu += PCU.get(kat, 0.0)

            dets.append({
                "label": kat,                   # label kategori (car/motorcycle/dll)
                "box_xyxy": [x1, y1, x2, y2],   # bounding box dalam format xyxy
            })

    return dets, agg_counts, agg_pcu


def process_image_bytes(img_bytes, model_type: str):
    arr = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None

    model_type = (model_type or "yolo").lower()
    if model_type == "fcos":
        dets, agg_counts, agg_pcu = detect_fcos(bgr)
    elif model_type == "rtdetr":
        dets, agg_counts, agg_pcu = detect_rtdetr(bgr)
    else:
        # default: YOLO
        dets, agg_counts, agg_pcu = detect_yolo(bgr)

    overlay = draw_overlay(bgr, dets, agg_counts, agg_pcu)
    return overlay, agg_counts, agg_pcu

# ====================== FUZZY ======================
def fuzzy_low(x):
    if x <= 0:
        return 1.0
    if 0 < x <= 15:
        return 1 - (x / 15.0)
    if 15 < x <= 25:
        return max(0.0, (25 - x) / 10.0)
    return 0.0

def fuzzy_med(x):
    if 10 < x <= 20:
        return (x - 10) / 10.0
    if 20 < x <= 30:
        return (30 - x) / 10.0
    return 0.0

def fuzzy_high(x):
    if x <= 20:
        return 0.0
    if 20 < x <= 30:
        return (x - 20) / 10.0
    return 1.0

def compute_fuzzy(df):
    BASE = 10.0
    EXTRA = 40.0
    G_MIN = MIN_GREEN_FUZZY
    G_MAX = MAX_GREEN_FUZZY

    weights = {}
    for idx, row in df.iterrows():
        p = row["PCU_total"]
        w = 0.5 * fuzzy_low(p) + 1.0 * fuzzy_med(p) + 1.5 * fuzzy_high(p)
        weights[idx] = max(w, 0.1)

    SW = sum(weights.values())
    green = {}
    for idx in df.index:
        share = weights[idx] / SW
        g = BASE + share * EXTRA
        g = max(G_MIN, min(G_MAX, g))
        green[idx] = g

    rows = []
    for idx in df.index:
        g = green[idx]
        r = sum(green[j] for j in green if j != idx)
        rows.append(
            {
                "Persimpangan": idx,
                "PCU_total": df.loc[idx]["PCU_total"],
                "Green_time": round(g, 2),
                "Red_time": round(r, 2),
            }
        )

    return pd.DataFrame(rows).set_index("Persimpangan")

# ====================== ROUTES ======================
@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
        },
    )

@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    model_type: str = Form("yolo"),
    utara: UploadFile = File(...),
    timur: UploadFile = File(...),
    selatan: UploadFile = File(...),
    barat: UploadFile = File(...),
):
    """
    model_type: "yolo" | "fcos" | "rtdetr"
    """
    global LAST_FUZZY

    intersections = {
        "UTARA": utara,
        "TIMUR": timur,
        "SELATAN": selatan,
        "BARAT": barat,
    }

    output_paths = {}
    rows = []

    for name, file in intersections.items():
        img_bytes = await file.read()

        out, counts, pcu = process_image_bytes(img_bytes, model_type=model_type)
        if out is None:
            continue

        out_dir = "static/output"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{name}.jpg"
        cv2.imwrite(out_path, out)
        output_paths[name] = "/" + out_path

        rows.append(
            {
                "Persimpangan": name,
                "PCU_total": round(pcu, 2),
                "car": int(counts["car"]),
                "motorcycle": int(counts["motorcycle"]),
                "bicycle": int(counts["bicycle"]),
                "kendaraan_besar": int(counts["kendaraan_besar"]),
            }
        )

    df_pcu = pd.DataFrame(rows).set_index("Persimpangan")
    df_fuzzy = compute_fuzzy(df_pcu)

    new_last = {}
    for arah in ["UTARA", "TIMUR", "SELATAN", "BARAT"]:
        if arah in df_fuzzy.index:
            new_last[arah] = {
                "Green_time": float(df_fuzzy.loc[arah, "Green_time"]),
                "Red_time": float(df_fuzzy.loc[arah, "Red_time"]),
            }
        else:
            new_last[arah] = LAST_FUZZY.get(
                arah, {"Green_time": 10.0, "Red_time": 50.0}
            )

    LAST_FUZZY = new_last

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": {
                "images": output_paths,
                "pcu_table": df_pcu.to_dict(orient="index"),
                "fuzzy_table": df_fuzzy.to_dict(orient="index"),
                "model_type": model_type,
            },
        },
    )

@app.get("/pico_state", response_class=PlainTextResponse)
def pico_state():
    global LAST_FUZZY

    def g(a):
        return float(LAST_FUZZY.get(a, {}).get("Green_time", 10.0))

    def r(a):
        return float(LAST_FUZZY.get(a, {}).get("Red_time", 50.0))

    gU = g("UTARA")
    rU = r("UTARA")
    gT = g("TIMUR")
    rT = r("TIMUR")
    gS = g("SELATAN")
    rS = r("SELATAN")
    gB = g("BARAT")
    rB = r("BARAT")

    payload = f"{gU:.2f},{rU:.2f},{gT:.2f},{rT:.2f},{gS:.2f},{rS:.2f},{gB:.2f},{rB:.2f}"
    return payload
