from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ExifTags
import io
from ultralytics import YOLO
import uvicorn
import numpy as np
import cv2
import base64


app = FastAPI(title="ALPR - License Plate Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load YOLOv8 license plate detector with safe loading
import torch
import os

# Monkey patch torch.load to disable weights_only for this trusted model
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

model = YOLO("license_plate_detector.pt")

# Mount static web app at /web
app.mount("/web", StaticFiles(directory="webapp", html=True), name="web")

# Allow very large images (disable PIL DecompressionBomb protection)
Image.MAX_IMAGE_PIXELS = None

# Optional: enable HEIC/HEIF support if available
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass


@app.get("/")
def index_redirect():
    return RedirectResponse(url="/web/")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Robust load with PIL (handles orientation, HEIC if enabled), fallback to OpenCV
    image = None
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        # Correct orientation via EXIF if present
        try:
            exif = pil._getexif()
            if exif is not None:
                orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
                if orientation_key and orientation_key in exif:
                    orientation = exif[orientation_key]
                    if orientation == 3:
                        pil = pil.rotate(180, expand=True)
                    elif orientation == 6:
                        pil = pil.rotate(270, expand=True)
                    elif orientation == 8:
                        pil = pil.rotate(90, expand=True)
        except Exception:
            pass
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        image = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        try:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            return JSONResponse({"error": "Unsupported or corrupted image format"}, status_code=415)

    if image is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Optional server-side resize for very large images
    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side > 2000:
        scale = 1600.0 / max_side
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Run model with timing
    t0 = cv2.getTickCount()
    results = model.predict(source=image, verbose=False)
    t1 = cv2.getTickCount()
    time_ms = (t1 - t0) / cv2.getTickFrequency() * 1000.0
    result = results[0]

    boxes = []
    if hasattr(result, "boxes") and result.boxes is not None:
        for b in result.boxes:
            xyxy = b.xyxy.cpu().numpy().tolist()[0]
            conf = float(b.conf.cpu().numpy().tolist()[0]) if hasattr(b, "conf") else None
            boxes.append({"xyxy": xyxy, "confidence": conf})

    # Blur only plate text area: shrink box slightly, apply strong Gaussian blur within
    h_out, w_out = image.shape[:2]
    output = image.copy()
    shrink_ratio = 0.06  # very tight crop baseline

    for box in boxes:
        x1f, y1f, x2f, y2f = box["xyxy"]
        x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # start with an inward shrink
        sx = int(bw * shrink_ratio)
        sy = int(bh * shrink_ratio)
        xa = min(max(0, x1 + sx), w_out - 1)
        ya = min(max(0, y1 + sy), h_out - 1)
        xb = max(min(w_out, x2 - sx), xa + 1)
        yb = max(min(h_out, y2 - sy), ya + 1)

        # refine to plate-like aspect ratio with fixed 30px minimum height for consistent vertical coverage
        plate_ar = 2.5
        cur_w = xb - xa
        cur_h = yb - ya
        if cur_w > 0 and cur_h > 0:
            # compute desired dimensions centered within current box
            desired_h = max(50, min(cur_h, int(cur_w / plate_ar)))  # fixed 50px minimum height
            desired_w = max(18, min(cur_w, int(desired_h * plate_ar)))
            cx = xa + cur_w // 2
            cy = ya + cur_h // 2
            xa = max(0, cx - desired_w // 2)
            ya = max(0, cy - desired_h // 2)
            xb = min(w_out, xa + desired_w)
            yb = min(h_out, ya + desired_h)

        if xa < xb and ya < yb:
            roi = output[ya:yb, xa:xb]
            rh, rw = roi.shape[:2]
            # HEAVY anonymization: strong pixelation inside the refined ROI
            small_w = max(1, int(max(rw * 0.06, 12)))
            small_h = max(1, int(max(rh * 0.06, 6)))
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            output[ya:yb, xa:xb] = pixelated

    _, buf = cv2.imencode('.jpg', output)
    img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    return {"image_base64": img_b64}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


