"""
Microplastic Detection Internal Microservice for Vercel Services
Provides internal YOLOv8 inference endpoints.
"""

import os
import io
import sys
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

# Apply PyTorch/Torchvision compatibility patch for missing C++ extension ops
try:
    import torch
    import torchvision
    import torchvision.ops

    for _op in ["nms", "qnms"]:
        try:
            torch.library.define(f"torchvision::{_op}", "(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        except Exception:
            pass

    def _pure_torch_nms(boxes, scores, iou_threshold):
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break
            xx1 = torch.maximum(x1[i], x1[order[1:]])
            yy1 = torch.maximum(y1[i], y1[order[1:]])
            xx2 = torch.minimum(x2[i], x2[order[1:]])
            yy2 = torch.minimum(y2[i], y2[order[1:]])
            w = torch.clamp(xx2 - xx1, min=0)
            h = torch.clamp(yy2 - yy1, min=0)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
            order = order[ids + 1]
        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

    torchvision.ops.nms = _pure_torch_nms
    if hasattr(torchvision.ops, "boxes"):
        torchvision.ops.boxes.nms = _pure_torch_nms
except Exception:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import cv2
except ImportError:
    cv2 = None

app = Flask(__name__)
CORS(app)

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATHS = [
    os.path.join(current_dir, "best.pt"),
    os.path.join(current_dir, "yolov8n.pt"),
    os.path.join(current_dir, "yolo11n.pt"),
    os.path.join(current_dir, "yolov8s.pt")
]

model = None
model_loaded_path = None

def init_model():
    global model, model_loaded_path
    if YOLO is None:
        return
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = YOLO(p)
                model_loaded_path = p
                print(f"Loaded YOLO model from {p}")
                break
            except Exception as e:
                print(f"Failed loading {p}: {e}")

init_model()

def apply_clahe_enhancement(pil_img):
    if cv2 is None:
        return pil_img
    img_np = np.array(pil_img)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)

def classify_morphology(aspect_ratio, area_px):
    if aspect_ratio >= 3.0 or aspect_ratio <= 0.33:
        return "Fiber"
    elif aspect_ratio >= 0.8 and aspect_ratio <= 1.25 and area_px >= 500:
        return "Pellet / Nurdle"
    elif area_px > 1500 and (aspect_ratio < 0.8 or aspect_ratio > 1.25):
        return "Film"
    else:
        return "Fragment"

def classify_size_band(max_dim_um):
    if max_dim_um < 1.0:
        return "Nanoplastic (<1 µm)"
    elif max_dim_um <= 100.0:
        return "Small Microplastic (1–100 µm)"
    elif max_dim_um <= 1000.0:
        return "Large Microplastic (100–1000 µm)"
    else:
        return "Mesoplastic (>1000 µm)"

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "microplastic-detection",
        "model_loaded": model is not None,
        "model_path": model_loaded_path,
        "clahe_available": cv2 is not None
    })

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files and "image_base64" not in request.form:
        return jsonify({"error": "No image provided"}), 400

    try:
        conf_thresh = float(request.form.get("confidence", 0.25))
        um_per_px = float(request.form.get("um_per_px", 1.5))
        apply_clahe = request.form.get("apply_clahe", "false").lower() == "true"

        if "image" in request.files:
            file = request.files["image"]
            img_bytes = file.read()
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif "image_base64" in request.form:
            raw_b64 = request.form["image_base64"].split(",")[-1]
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            return jsonify({"error": "Invalid image payload"}), 400

        if model is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded on internal service",
                "message": "Local YOLO model weights are not available on serverless runtime."
            }), 503

        width, height = pil_img.size
        processed_img = apply_clahe_enhancement(pil_img) if apply_clahe else pil_img
        inference_imgsz = 1280 if max(width, height) >= 1200 else 640

        results = model.predict(processed_img, conf=conf_thresh, imgsz=inference_imgsz)
        result = results[0]

        boxes_data = []
        high_conf_count = 0
        total_conf = 0.0

        morphology_counts = {"Fiber": 0, "Fragment": 0, "Pellet / Nurdle": 0, "Film": 0}
        size_band_counts = {
            "Nanoplastic (<1 µm)": 0,
            "Small Microplastic (1–100 µm)": 0,
            "Large Microplastic (100–1000 µm)": 0,
            "Mesoplastic (>1000 µm)": 0
        }

        if result.boxes is not None and len(result.boxes) > 0:
            for idx, box in enumerate(result.boxes):
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                c = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                raw_label = result.names.get(cls_id, "microplastic")

                x1, y1, x2, y2 = xyxy
                box_w_px = x2 - x1
                box_h_px = y2 - y1
                area_px = box_w_px * box_h_px

                width_um = box_w_px * um_per_px
                height_um = box_h_px * um_per_px
                area_um2 = area_px * (um_per_px ** 2)
                max_dim_um = max(width_um, height_um)

                aspect_ratio = round(box_w_px / max(box_h_px, 0.1), 2)
                raw_roundness = (4 * np.pi * area_px) / max((2 * (box_w_px + box_h_px)) ** 2, 0.1)
                roundness = round(float(max(0.0, min(1.0, raw_roundness))), 3)

                morphology = classify_morphology(aspect_ratio, area_px)
                size_band = classify_size_band(max_dim_um)

                morphology_counts[morphology] = morphology_counts.get(morphology, 0) + 1
                size_band_counts[size_band] = size_band_counts.get(size_band, 0) + 1

                if c >= 0.5:
                    high_conf_count += 1
                total_conf += c

                boxes_data.append({
                    "id": idx + 1,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "width_px": round(box_w_px, 1),
                    "height_px": round(box_h_px, 1),
                    "area_px": round(area_px, 1),
                    "width_um": round(width_um, 1),
                    "height_um": round(height_um, 1),
                    "area_um2": round(area_um2, 1),
                    "confidence": round(c, 4),
                    "class": raw_label,
                    "morphology": morphology,
                    "size_band": size_band,
                    "aspect_ratio": aspect_ratio,
                    "roundness": roundness
                })

        particle_count = len(boxes_data)
        avg_confidence = round(total_conf / particle_count, 4) if particle_count > 0 else 0.0

        if particle_count == 0:
            risk = "Zero Particles Detected"
            risk_color = "#10b981"
        elif particle_count < 10:
            risk = "Low Contamination"
            risk_color = "#3b82f6"
        elif particle_count < 30:
            risk = "Moderate Contamination"
            risk_color = "#f59e0b"
        else:
            risk = "High Contamination Hazard"
            risk_color = "#ef4444"

        # Custom annotation — blue boxes with "microplastic 0.35" labels (new style)
        try:
            from PIL import ImageDraw, ImageFont
            annotated_pil = processed_img.copy()
            draw = ImageDraw.Draw(annotated_pil, "RGBA")

            BOX_COLOR = (59, 130, 246)        # #3B82F6 blue (RGB)
            LABEL_BG   = (0, 0, 0, 180)       # semi-transparent dark background
            LABEL_FG   = (255, 255, 255, 255) # white text
            lw = max(2, int(min(annotated_pil.width, annotated_pil.height) / 300))

            # Try to load a small font; fallback to default
            try:
                font_size = max(12, int(annotated_pil.width / 70))
                font = ImageFont.truetype("arial.ttf", font_size)
            except Exception:
                try:
                    font = ImageFont.load_default(size=max(12, int(annotated_pil.width / 70)))
                except Exception:
                    font = ImageFont.load_default()

            for d in boxes_data:
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]

                # Draw bounding box (blue)
                for t in range(lw):
                    draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=BOX_COLOR)

                # Label: "microplastic 0.35"
                label = f"{d['class'].lower()} {d['confidence']:.2f}"
                try:
                    bbox_t = font.getbbox(label)
                    text_w = bbox_t[2] - bbox_t[0]
                    text_h = bbox_t[3] - bbox_t[1]
                except Exception:
                    text_w, text_h = len(label) * 7, 14

                pad = 4
                label_h = text_h + pad * 2
                label_y = (y1 - label_h) if (y1 - label_h) >= 0 else y1 + 2

                # Dark background for label
                draw.rectangle(
                    [x1, label_y, x1 + text_w + pad * 2, label_y + label_h],
                    fill=LABEL_BG
                )
                # White label text
                draw.text((x1 + pad, label_y + pad), label, fill=LABEL_FG, font=font)

        except Exception as ann_err:
            print(f"Custom annotation fallback: {ann_err}")
            try:
                annotated_bgr = result.plot()
                annotated_rgb = annotated_bgr[..., ::-1]
                annotated_pil = Image.fromarray(annotated_rgb)
            except Exception:
                annotated_pil = processed_img

        buf = io.BytesIO()
        annotated_pil.save(buf, format="JPEG", quality=92)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "success": True,
            "particle_count": particle_count,
            "high_conf_count": high_conf_count,
            "avg_confidence": avg_confidence,
            "contamination_risk": risk,
            "risk_color": risk_color,
            "um_per_px": um_per_px,
            "clahe_applied": apply_clahe,
            "morphology_counts": morphology_counts,
            "size_band_counts": size_band_counts,
            "detections": boxes_data,
            "annotated_image": f"data:image/jpeg;base64,{base64_img}",
            "image_dimensions": {"width": width, "height": height},
            "engine_used": "YOLOv8 Internal Microservice"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
