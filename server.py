"""
Microplastic Detector — Flask API Backend
Serves the web front-end and provides AI detection API endpoint using YOLOv8.
"""

import os
import io
import sys
import base64
import re
import json
import requests  # pyrefly: ignore [missing-import] # type: ignore

# Ensure script directory is always in Python search path for clean module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # pyrefly: ignore [missing-import] # type: ignore
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory  # pyrefly: ignore [missing-import] # type: ignore
from flask_cors import CORS  # pyrefly: ignore [missing-import] # type: ignore
from PIL import Image  # pyrefly: ignore [missing-import] # type: ignore
import numpy as np  # pyrefly: ignore [missing-import] # type: ignore

from gemini_engine import (
    is_gemini_configured,
    detect_microplastics_with_gemini,
    train_gemini_dataset_context,
    get_gemini_api_key
)

# Apply PyTorch/Torchvision compatibility patch for missing C++ extension ops
try:
    import torch  # pyrefly: ignore [missing-import] # type: ignore
    import torchvision  # pyrefly: ignore [missing-import] # type: ignore
    import torchvision.ops  # pyrefly: ignore [missing-import] # type: ignore

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
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")

app = Flask(__name__, static_folder=WEB_DIR, static_url_path="")
CORS(app)

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
model = None

def load_yolo_model():
    global model
    if YOLO is None:
        print("⚠️ Ultralytics library not installed.")
        return
    if os.path.exists(MODEL_PATH):
        print(f"🧠 Loading custom YOLO model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("✅ Custom model loaded successfully!")
    elif os.path.exists(os.path.join(BASE_DIR, "yolov8n.pt")):
        fallback_path = os.path.join(BASE_DIR, "yolov8n.pt")
        print(f"🧠 Loading fallback model {fallback_path}...")
        model = YOLO(fallback_path)
    else:
        print("⚠️ No model file found.")

load_yolo_model()

def apply_clahe_enhancement(pil_img):
    """Apply Contrast Limited Adaptive Histogram Equalization to boost contrast for translucent plastics."""
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
    """Estimate particle morphology based on geometric aspect ratio and particle size."""
    if aspect_ratio >= 3.0 or aspect_ratio <= 0.33:
        return "Fiber"
    elif aspect_ratio >= 0.8 and aspect_ratio <= 1.25 and area_px >= 500:
        return "Pellet / Nurdle"
    elif area_px > 1500 and (aspect_ratio < 0.8 or aspect_ratio > 1.25):
        return "Film"
    else:
        return "Fragment"

def classify_size_band(max_dim_um):
    """Categorize particle physical dimension into scientific classification bands."""
    if max_dim_um < 1.0:
        return "Nanoplastic (<1 µm)"
    elif max_dim_um <= 100.0:
        return "Small Microplastic (1–100 µm)"
    elif max_dim_um <= 1000.0:
        return "Large Microplastic (100–1000 µm)"
    else:
        return "Mesoplastic (>1000 µm)"

@app.route("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

@app.route("/dashboard")
@app.route("/dashboard/")
def dashboard():
    return send_from_directory(WEB_DIR, "dashboard.html")

@app.route("/api/health", methods=["GET"])
def health():
    detection_service_url = os.environ.get("MICROPLASTIC_DETECTION_URL")
    return jsonify({
        "status": "online",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH if model else None,
        "clahe_available": cv2 is not None,
        "gemini_configured": is_gemini_configured(),
        "gemini_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "detection_service_bound": bool(detection_service_url),
        "detection_service_url": detection_service_url if detection_service_url else None
    })

@app.route("/api/train_gemini", methods=["POST"])
def train_gemini():
    """Endpoint to trigger in-context training and conditioning for Gemini model using local dataset."""
    res = train_gemini_dataset_context()
    return jsonify(res)

@app.route("/api/samples", methods=["GET"])
def samples():
    """Return list of sample test images available in the root directory."""
    sample_files = []
    if os.path.exists(BASE_DIR):
        for f in os.listdir(BASE_DIR):
            if f.startswith("test_image") and f.lower().endswith((".jpg", ".jpeg", ".png")):
                sample_files.append(f)
    sample_files.sort()
    return jsonify({"samples": sample_files})

@app.route("/sample_image/<filename>", methods=["GET"])
def get_sample_image(filename):
    """Serve sample images for quick testing."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(BASE_DIR, safe_name)
    if os.path.exists(file_path):
        return send_from_directory(BASE_DIR, safe_name)
    return jsonify({"error": "File not found"}), 404

def process_single_detection(pil_img, conf_thresh, apply_clahe, um_per_px):
    """Perform object detection, micrometer scaling, and morphological profiling on an image."""
    width, height = pil_img.size

    # Apply CLAHE contrast enhancement if requested
    processed_img = apply_clahe_enhancement(pil_img) if apply_clahe else pil_img

    # Dynamic image size selection
    inference_imgsz = 1280 if max(width, height) >= 1200 else 640

    # Run YOLO inference
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

            # Physical micrometer calculations
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

    # Contamination hazard risk assessment
    if particle_count == 0:
        risk = "Zero Particles Detected"
        risk_color = "#10b981" # green
    elif particle_count < 10:
        risk = "Low Contamination"
        risk_color = "#3b82f6" # blue
    elif particle_count < 30:
        risk = "Moderate Contamination"
        risk_color = "#f59e0b" # amber
    else:
        risk = "High Contamination Hazard"
        risk_color = "#ef4444" # red

    # Render annotated visualization image — new style: blue boxes, "microplastic 0.35" labels
    try:
        # Blue box colour in BGR for OpenCV (= #3B82F6 → R59 G130 B246)
        annotated_bgr = result.plot(
            boxes=True,
            conf=True,
            labels=True,
            line_width=2,
            font_size=12,
            color=(246, 130, 59),   # BGR for #3B82F6
        )
        annotated_rgb = annotated_bgr[..., ::-1]
        annotated_pil = Image.fromarray(annotated_rgb)
    except Exception as pe:
        print(f"Visualization plot fallback: {pe}")
        annotated_pil = processed_img

    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=92)
    base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
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
        "image_dimensions": {"width": width, "height": height}
    }

@app.route("/api/detect", methods=["POST"])
def detect():
    if "image" not in request.files and "sample" not in request.form:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        conf_thresh = float(request.form.get("confidence", 0.25))
        um_per_px = float(request.form.get("um_per_px", 1.5))
        apply_clahe = request.form.get("apply_clahe", "false").lower() == "true"
        engine = request.form.get("engine", "hybrid").lower()
        filename = "uploaded_image.jpg"

        if "image" in request.files:
            file = request.files["image"]
            if not file or not file.filename:
                return jsonify({"error": "Selected file is empty or missing"}), 400
            filename = os.path.basename(file.filename)
            img_bytes = file.read()
            if not img_bytes:
                return jsonify({"error": "Uploaded image file is empty"}), 400
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                return jsonify({"error": "Invalid or unreadable image file format"}), 400
        elif "sample" in request.form:
            sample_name = os.path.basename(request.form.get("sample", ""))
            filename = sample_name
            sample_path = os.path.join(BASE_DIR, sample_name)
            if not sample_name or not os.path.exists(sample_path):
                return jsonify({"error": f"Sample image '{sample_name}' not found"}), 404
            try:
                pil_img = Image.open(sample_path).convert("RGB")
            except Exception:
                return jsonify({"error": "Failed to load sample image file"}), 400
        else:
            return jsonify({"error": "Invalid request"}), 400

        # Run detection based on selected AI engine
        res_data = None
        if engine == "gemini":
            res_data = detect_microplastics_with_gemini(pil_img, conf_thresh, um_per_px)
            gemini_model_used = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
            res_data["success"] = True
            res_data["filename"] = filename
            if "engine_used" not in res_data:
                res_data["engine_used"] = res_data.get("ai_provider", f"Google Gemini AI ({gemini_model_used})")
            return jsonify(res_data)

        # Check if internal microplastic-detection service is bound via Vercel
        detection_service_url = os.environ.get("MICROPLASTIC_DETECTION_URL")
        if detection_service_url and engine != "gemini":
            try:
                target_url = f"{detection_service_url.rstrip('/')}/api/detect"
                img_io = io.BytesIO()
                pil_img.save(img_io, format="JPEG", quality=95)
                img_io.seek(0)
                forward_data = {
                    "confidence": conf_thresh,
                    "um_per_px": um_per_px,
                    "apply_clahe": str(apply_clahe).lower()
                }
                files = {"image": (filename, img_io, "image/jpeg")}
                resp = requests.post(target_url, data=forward_data, files=files, timeout=45)
                if resp.status_code == 200:
                    cand = resp.json()
                    if cand.get("particle_count") is not None:
                        res_data = cand
                        res_data["filename"] = filename
                        if engine == "hybrid":
                            try:
                                gemini_enrichment = detect_microplastics_with_gemini(pil_img, conf_thresh, um_per_px)
                                if "ai_analysis" in gemini_enrichment:
                                    res_data["ai_analysis"] = gemini_enrichment["ai_analysis"]
                                    res_data["ai_provider"] = "Hybrid Dual Engine (Internal YOLOv8 + Gemini AI)"
                                    res_data["engine_used"] = "Hybrid Dual Engine (Internal YOLOv8 + Gemini AI)"
                            except Exception as ge:
                                print(f"Hybrid Gemini enrichment skipped: {ge}")
                        return jsonify(res_data)
                print(f"Internal detection service returned {resp.status_code}, falling back...")
            except Exception as fe:
                print(f"Internal detection service request failed: {fe}, falling back...")

        # Standard local YOLO detection fallback if weights file exists
        if model is not None:
            res_data = process_single_detection(pil_img, conf_thresh, apply_clahe, um_per_px)
            res_data["success"] = True
            res_data["filename"] = filename
            res_data["engine_used"] = "YOLOv8 Custom Computer Vision"

            if engine == "hybrid":
                try:
                    gemini_enrichment = detect_microplastics_with_gemini(pil_img, conf_thresh, um_per_px)
                    if "ai_analysis" in gemini_enrichment:
                        res_data["ai_analysis"] = gemini_enrichment["ai_analysis"]
                        res_data["ai_provider"] = "Hybrid Dual Engine (YOLOv8 + Gemini AI)"
                        res_data["engine_used"] = "Hybrid Dual Engine (YOLOv8 + Gemini AI)"
                except Exception as ge:
                    print(f"Hybrid Gemini enrichment skipped: {ge}")

            return jsonify(res_data)

        # Resilient AI & Computer Vision pipeline (detect_microplastics_with_gemini auto-falls back to optical if needed)
        res_data = detect_microplastics_with_gemini(pil_img, conf_thresh, um_per_px)
        res_data["success"] = True
        res_data["filename"] = filename
        if "engine_used" not in res_data:
            res_data["engine_used"] = res_data.get("ai_provider", "MicroDetect Hybrid Vision Engine")
        return jsonify(res_data)

    except Exception as e:
        print(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch_detect", methods=["POST"])
def batch_detect():
    """Process multiple sample images in batch mode."""
    detection_service_url = os.environ.get("MICROPLASTIC_DETECTION_URL")
    if model is None and not is_gemini_configured() and not detection_service_url:
        return jsonify({"error": "No detection engine available (YOLO not loaded & Gemini not configured)"}), 500
    if "images" not in request.files:
        return jsonify({"error": "No images provided for batch detection"}), 400

    try:
        conf_thresh = float(request.form.get("confidence", 0.25))
        um_per_px = float(request.form.get("um_per_px", 1.5))
        apply_clahe = request.form.get("apply_clahe", "false").lower() == "true"

        files = request.files.getlist("images")
        results_list = []
        total_particles_all = 0

        for file in files:
            try:
                img_bytes = file.read()
                if detection_service_url:
                    try:
                        target_url = f"{detection_service_url.rstrip('/')}/api/detect"
                        resp = requests.post(
                            target_url,
                            data={"confidence": conf_thresh, "um_per_px": um_per_px, "apply_clahe": str(apply_clahe).lower()},
                            files={"image": (file.filename, io.BytesIO(img_bytes), "image/jpeg")},
                            timeout=30
                        )
                        single_res = resp.json() if resp.status_code == 200 else None
                    except Exception as b_fe:
                        print(f"Batch internal service error for {file.filename}: {b_fe}")
                        single_res = None
                else:
                    single_res = None

                if single_res is None:
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    if model is not None:
                        single_res = process_single_detection(pil_img, conf_thresh, apply_clahe, um_per_px)
                    elif is_gemini_configured():
                        single_res = detect_microplastics_with_gemini(pil_img, conf_thresh, um_per_px)
                    else:
                        print(f"Batch skip {file.filename}: no engine available")
                        continue
                single_res["filename"] = file.filename
                results_list.append(single_res)
                total_particles_all += single_res.get("particle_count", 0)
            except Exception as fe:
                print(f"Batch file error on {file.filename}: {fe}")

        return jsonify({
            "success": True,
            "sample_count": len(results_list),
            "total_particles": total_particles_all,
            "batch_results": results_list
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat_assistant():
    """AI Assistant endpoint for answering microplastics, environmental science, coding, and general knowledge queries."""
    user_msg = ""
    try:
        req = request.get_json(force=True, silent=True) or {}
        user_msg = str(req.get("message", "")).strip()
        custom_key = str(req.get("api_key", "")).strip()
        history = req.get("history", [])
        scan_ctx = req.get("scan_context") or {}

        if not user_msg:
            return jsonify({"reply": "Hello! 👋 I am MicroDetect AI Assistant. Ask me ANY question — about microplastics, WHO guidelines, your active scan, polymer science, coding, or math!"})

        # Build active scan prompt context if available
        scan_summary_str = ""
        if isinstance(scan_ctx, dict) and scan_ctx.get("particle_count") is not None:
            scan_summary_str = (
                f"\n--- ACTIVE SAMPLE SCAN DATA ---\n"
                f"• Filename: {scan_ctx.get('filename', 'Current Sample')}\n"
                f"• Particles Detected: {scan_ctx.get('particle_count', 0)}\n"
                f"• High Confidence Count: {scan_ctx.get('high_conf_count', 0)}\n"
                f"• Avg Confidence: {round(float(scan_ctx.get('avg_confidence', 0))*100, 1)}%\n"
                f"• Contamination Hazard Risk: {scan_ctx.get('contamination_risk', 'Unknown')}\n"
                f"• Morphology Breakdown: {json.dumps(scan_ctx.get('morphology_counts', {}))}\n"
                f"• Size Band Breakdown: {json.dumps(scan_ctx.get('size_band_counts', {}))}\n"
                f"--------------------------------\n"
            )

        # 1. Try Gemini AI LLM if API Key is provided via request or .env
        gemini_key = custom_key if (custom_key and len(custom_key) > 15 and "YOUR_GEMINI" not in custom_key) else get_gemini_api_key()

        if gemini_key and len(gemini_key) > 15 and "YOUR_GEMINI_API_KEY" not in gemini_key:
            try:
                system_prompt = (
                    "You are MicroDetect AI Assistant, an expert, friendly conversational AI for Environmental Science, Microplastics Detection, Computer Vision, Chemistry, Math, and General Knowledge.\n"
                    "Provide accurate, clear, and well-formatted answers using markdown (bolding, lists, code blocks, tables).\n"
                    f"{scan_summary_str}"
                )

                contents_list = []
                # Include previous conversation turn history if available
                if isinstance(history, list) and len(history) > 0:
                    for turn in history[-6:]: # include up to last 6 turns
                        role = "user" if turn.get("role") == "user" else "model"
                        txt = str(turn.get("text", "")).strip()
                        if txt:
                            contents_list.append({"role": role, "parts": [{"text": txt}]})

                # Append current user prompt with system context
                contents_list.append({"role": "user", "parts": [{"text": f"{system_prompt}\nUser Query: {user_msg}"}]})

                payload = {"contents": contents_list}
                chat_models = ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"]
                for m in chat_models:
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={gemini_key}"
                        r = requests.post(url, json=payload, timeout=20)
                        if r.status_code == 200:
                            res_json = r.json()
                            candidates = res_json.get("candidates", [])
                            if candidates and "content" in candidates[0]:
                                parts = candidates[0]["content"].get("parts", [])
                                if parts and "text" in parts[0]:
                                    return jsonify({"reply": parts[0]["text"].strip()})
                    except Exception as me:
                        print(f"Chat model {m} attempt skipped: {me}")
            except Exception as ge:
                print(f"Gemini LLM chat exception: {ge}")

        # 2. Advanced Knowledge & Offline Smart AI Engine
        m = user_msg.lower()

        # Check if user asks about active scan
        if any(w in m for w in ["active scan", "my sample", "explain my scan", "current scan", "scan summary", "my results"]):
            if isinstance(scan_ctx, dict) and scan_ctx.get("particle_count") is not None:
                cnt = scan_ctx.get("particle_count", 0)
                risk = scan_ctx.get("contamination_risk", "Unknown Risk")
                high_c = scan_ctx.get("high_conf_count", 0)
                avg_c = round(float(scan_ctx.get("avg_confidence", 0))*100, 1)
                fname = scan_ctx.get("filename", "Uploaded Sample")
                who_status = "🔴 Exceeding WHO drinking water safe limit (<10 particles)." if cnt > 10 else "🟢 Within safe WHO drinking water parameters (<10 particles)."
                
                reply = (
                    f"📊 **Active Sample Scan Summary**:\n"
                    f"• **Sample File**: `{fname}`\n"
                    f"• **Particles Detected**: **{cnt}** microplastic particles\n"
                    f"• **High Confidence Detections**: **{high_c}** (>50% certainty)\n"
                    f"• **Average AI Certainty**: **{avg_c}%**\n"
                    f"• **Contamination Hazard Level**: **{risk}**\n"
                    f"• **WHO Limit Assessment**: {who_status}"
                )
                return jsonify({"reply": reply})

        # Math evaluation handler (e.g., 25 * 4, 100 / 5, 45 + 55)
        math_match = re.search(r'^\s*(?:what\s+is\s+)?([\d\.\s\+\-\*\/\(\)]+)\s*\??$', m)
        if math_match:
            expr = math_match.group(1).strip()
            if any(op in expr for op in ['+', '-', '*', '/']) and re.match(r'^[\d\.\s\+\-\*\/\(\)]+$', expr):
                try:
                    result = eval(expr, {"__builtins__": None}, {})
                    return jsonify({"reply": f"🔢 **Math Calculation Result**:\n`{expr}` = **{result}**"})
                except Exception:
                    pass

        # Greetings & Persona
        if any(re.search(rf'\b{w}\b', m) for w in ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
            reply = "Hello! 👋 I'm your MicroDetect AI Assistant. Ask me **ANY question** on science, math, technology, microplastics detection, WHO guidelines, or active scan results!"
        elif any(w in m for w in ["who created", "who made", "who are you", "what are you"]):
            reply = "🤖 I am **MicroDetect AI Assistant**, an intelligent multi-domain virtual researcher built using YOLOv8 deep learning and Google Gemini AI vision models."
        elif any(w in m for w in ["thank", "thanks", "awesome", "great", "cool"]):
            reply = "You're very welcome! 🌟 Let me know if you have any more questions!"

        # General Science & Technology
        elif any(w in m for w in ["python", "flask", "javascript", "html", "css", "code", "programming"]):
            reply = "💻 **Programming & Software Engineering**:\n• **Flask**: A lightweight Python WSGI web app framework powering backend endpoints.\n• **YOLOv8**: Ultra-fast object detection library by Ultralytics.\n• **JavaScript (ES6+)**: Powers client-side interactivity, DOM manipulation, and charts."
        elif any(re.search(rf'\b{w}\b', m) for w in ["ai", "artificial intelligence", "deep learning", "machine learning"]):
            reply = "🧠 **Artificial Intelligence**: Machine Learning enables algorithms to learn patterns from datasets. Deep learning neural networks (like YOLOv8 and Gemini AI) use multi-layered artificial neurons for complex vision and natural language processing tasks."

        # Polymer Types & Chemistry
        elif any(re.search(rf'\b{w}\b', m) for w in ["pet", "polyethylene terephthalate"]):
            reply = "🧪 **PET (Polyethylene Terephthalate)** is a common thermoplastic resin used in single-use beverage bottles and synthetic fibers. When degraded by UV light or mechanical friction, it forms rigid, sharp-edged microplastic fragments."
        elif any(re.search(rf'\b{w}\b', m) for w in ["pe", "polyethylene", "hdpe", "ldpe"]):
            reply = "🧪 **Polyethylene (PE)** is the most widely produced polymer, used in plastic bags, food film, and packaging. It degrades into lightweight films and micro-fragments that float in aquatic environments."
        elif any(re.search(rf'\b{w}\b', m) for w in ["pp", "polypropylene"]):
            reply = "🧪 **Polypropylene (PP)** is a tough, heat-resistant polymer used in bottle caps, food containers, and automotive parts. PP microplastics are frequently detected as low-density floating particles."
        elif any(re.search(rf'\b{w}\b', m) for w in ["ps", "polystyrene", "styrofoam"]):
            reply = "🧪 **Polystyrene (PS)** is widely used in expanded foam packaging and disposable cutlery. It breaks down easily into brittle micro-spheres and toxic styrene monomers."
        elif any(re.search(rf'\b{w}\b', m) for w in ["pvc", "polyvinyl chloride"]):
            reply = "🧪 **Polyvinyl Chloride (PVC)** is a dense, rigid polymer used in piping and construction. PVC microplastics often contain plasticizer additives like phthalates."
        elif any(w in m for w in ["nylon", "polyester", "synthetic textile"]):
            reply = "🧵 **Synthetic Textiles (Nylon/Polyester)**: A single laundry cycle of synthetic garments can release over 700,000 micro-fibers into wastewater streams."

        # Health Risks & Toxicity
        elif any(w in m for w in ["health risk", "toxicity", "human body", "ingest microplastics", "disease risk"]):
            reply = "⚠️ **Human Health Impact**: Ingested microplastics (<10 µm) can cross intestinal barriers, enter bloodstream circulatory systems, and accumulate in liver/kidney tissues. They carry toxic additives (phthalates, BPA, heavy metals) causing cellular oxidative stress and endocrine disruption."
        elif any(w in m for w in ["cancer risk", "cellular toxicity"]):
            reply = "🧬 **Cellular Effects**: Studies show microplastics can trigger localized inflammatory responses, cellular toxicity, and oxidative stress in human tissues. Sub-micron nanoplastics (<1 µm) pose the highest risk due to cellular membrane penetration."

        # Environmental Impact & EVS Concepts
        elif any(w in m for w in ["marine ecosystem", "ocean pollution", "fish ingestion", "plankton"]):
            reply = "🐠 **Marine Ecosystem Impact**: Microplastics are ingested by plankton, fish, and marine mammals, causing digestive blockages, starvation, and bioaccumulation up the food marine trophic chain to human seafood consumers."
        elif any(w in m for w in ["microplastics environment", "environmental science", "evs"]):
            reply = "🌍 **Environmental Science (EVS)**: Over 14 million metric tons of microplastics sit on the global seafloor. Research by WHO (2019) revealed microplastic contamination in 93% of bottled drinking water samples globally."

        # Water Testing & WHO Limits
        elif any(re.search(rf'\b{w}\b', m) for w in ["who", "limit", "guideline", "standard", "safe", "drinking"]):
            reply = "⚠️ **WHO Water Safety Thresholds**:\n• **< 10 particles/sample**: Safe drinking water limit\n• **10 – 30 particles**: Moderate contamination warning\n• **> 30 particles**: Critical WHO hazard level requiring immediate filtration"
        elif any(w in m for w in ["how to test", "test a sample", "scan an image", "how to scan"]):
            reply = "To test a sample:\n1. Upload a microscopic sample image or select a Quick Test Sample.\n2. Click '🔬 Run AI Detection'.\n3. View particle counts, micrometer size breakdown, Risk Speedometer Gauge, and auto-generated Lab Report!"
        elif any(w in m for w in ["filter", "remediate", "treatment", "remove", "clean", "purify"]):
            reply = "💧 **Remediation & Filtration**: Effective filtration methods include Membrane Ultrafiltration (0.1 µm), Reverse Osmosis (RO), Granular Activated Carbon (GAC), and Membrane Bioreactors (MBR) to trap micro/nano particles."

        # Particle Morphologies & Sizes
        elif "fibers vs fragments" in m or ("fiber" in m and "fragment" in m):
            reply = "• 🧵 **Fibers**: Thin, elongated synthetic threads (Aspect Ratio ≥ 3.0) from textile washing cycles.\n• 🪨 **Fragments**: Jagged, rigid pieces from broken bottles/containers."
        elif any(re.search(rf'\b{w}\b', m) for w in ["fiber", "fragment", "nurdle", "film", "morphology"]):
            reply = "Morphological categories indicate origins:\n• 🧵 **Fibers**: From synthetic clothing laundry.\n• 🪨 **Fragments**: From rigid plastic degradation.\n• ⚪ **Pellets/Nurdles**: Pre-production industrial resin granules.\n• 🗞️ **Films**: From bags and packaging wraps."
        elif any(re.search(rf'\b{w}\b', m) for w in ["nanoplastic", "microplastic", "micrometer", "µm"]):
            reply = "📏 **Scientific Size Bands**:\n• **Nano**: < 1 µm (High cell penetration risk)\n• **Micro**: 1 – 10 µm\n• **Small**: 10 – 100 µm\n• **Medium**: 100 – 500 µm\n• **Large**: > 500 µm"

        # AI Detection & Computer Vision
        elif any(w in m for w in ["yolo", "v8", "neural network", "computer vision"]):
            reply = "🤖 **YOLOv8 AI Detection**: Our custom YOLOv8 neural network detects microscopic particles in milliseconds, drawing bounding boxes, predicting confidence scores, and calculating physical area (µm²) and dimensions."
        elif any(w in m for w in ["confidence score", "ai certainty"]):
            reply = "📊 **AI Confidence Score**: Indicates statistical certainty of classification (0% to 100%). Scores > 50% represent high-confidence definitive classifications."
        elif any(w in m for w in ["clahe", "contrast enhancement"]):
            reply = "⚡ **CLAHE Contrast**: Contrast Limited Adaptive Histogram Equalization enhances local image contrast to reveal translucent plastic particles visible under optical microscopes."
        elif any(w in m for w in ["micrometer scale", "calibration", "pixel scale"]):
            reply = "📏 **Micrometer Calibration**: Converts pixel distances into physical dimensions (µm). Use scale presets (e.g. 1.5 µm/px) or enter custom scale values in the calibration sidebar."

        # Website Features & Platform Guide
        elif any(w in m for w in ["report", "pdf", "lab report"]):
            reply = "📝 **Lab Report**: Under the '📝 Lab Report' tab, you can view an 8-section scientific report complete with executive summaries, particle breakdowns, and download it as a print-ready PDF."
        elif any(w in m for w in ["export", "csv", "json"]):
            reply = "📥 **Export Data**: Click the CSV or JSON buttons above the results table to download raw micrometer dimensions, confidence scores, and bounding box coordinates for external scientific software."
        elif any(w in m for w in ["batch", "multiple"]):
            reply = "📦 **Batch Analysis**: Switch to the '📦 Batch Analysis' tab to process multiple microscopic images simultaneously in parallel."
        elif any(w in m for w in ["history", "trend", "past", "log"]):
            reply = "📅 **Scan History**: The '📅 Scan History' tab logs all session scans, renders a real-time particle trend graph, and lets you reload any previous scan with one click."

        # Open-domain fallback response + Gemini API Key activation guide
        else:
            reply = (
                f"💡 **MicroDetect AI Assistant Response**:\n"
                f"I processed your query: *\"{user_msg}\"*\n\n"
                f"🔑 **To unlock 100% UNLIMITED answers for literally ANY question on any topic** (general knowledge, coding, math, history, science, etc.):\n"
                f"1. Get a **FREE Gemini API Key** from [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
                f"2. Add it into your `.env` file as `GEMINI_API_KEY=your_key_here` (or paste it in the AI Settings header icon above).\n"
                f"3. Once connected, your chatbot will answer **any question on the internet** using Google Gemini 2.0 Flash!"
            )

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"MicroDetect AI Assistant: I received your question '{user_msg}'. Ask me about microplastics, WHO safety limits, polymer chemistry, science, or enter your Gemini API Key for unlimited AI responses!"}), 200


if __name__ == "__main__":
    print("🚀 Microplastic Detection API Server starting on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)


