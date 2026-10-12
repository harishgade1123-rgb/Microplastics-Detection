"""
MicroDetect AI — Gemini Fine-Tuning Dataset Exporter
Generates JSONL dataset for fine-tuning Gemini vision models on Google AI Studio.
"""

import os
import sys
import json
import base64

# Force UTF-8 encoding for Windows stdout console printing
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

OUTPUT_JSONL = "gemini_finetune_dataset.jsonl"
SAMPLE_IMAGES = ["test_image.jpg", "test_image2.jpg", "test_image3.jpg", "test_image4.jpg", "test_image5.jpg", "test_image6.jpg"]

# Ground truth training targets for sample images
GROUND_TRUTH = {
    "test_image.jpg": {
        "particle_count": 11,
        "morphology_counts": {"Fiber": 6, "Fragment": 3, "Pellet / Nurdle": 2, "Film": 0},
        "size_band_counts": {"Nanoplastic (<1 µm)": 0, "Small Microplastic (1–100 µm)": 7, "Large Microplastic (100–1000 µm)": 4, "Mesoplastic (>1000 µm)": 0},
        "contamination_risk": "Moderate Contamination",
        "ai_analysis": "Water filtration sample containing synthetic textile fibers and irregular plastic fragments. High prevalence of small microplastics (1-100 µm)."
    },
    "test_image2.jpg": {
        "particle_count": 38,
        "morphology_counts": {"Fiber": 10, "Fragment": 18, "Pellet / Nurdle": 6, "Film": 4},
        "size_band_counts": {"Nanoplastic (<1 µm)": 2, "Small Microplastic (1–100 µm)": 20, "Large Microplastic (100–1000 µm)": 14, "Mesoplastic (>1000 µm)": 2},
        "contamination_risk": "High Contamination Hazard",
        "ai_analysis": "Severe microplastic accumulation detected in marine sediment extract. Predominantly jagged polypropylene fragments and industrial nurdles."
    }
}

def export_jsonl():
    records = []
    print("📦 Exporting ground-truth training records for Gemini Fine-Tuning...")
    
    for filename, gt in GROUND_TRUTH.items():
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            
            record = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"inline_data": {"mime_type": "image/jpeg", "data": b64_str}},
                            {"text": "Analyze this microscopic sample image for microplastic particles, particle count, size bands, morphology, and environmental hazard assessment. Return JSON."}
                        ]
                    },
                    {
                        "role": "model",
                        "parts": [
                            {"text": json.dumps(gt, indent=2)}
                        ]
                    }
                ]
            }
            records.append(record)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r) + "\n")
            
    print(f"✅ Generated {len(records)} training samples in '{OUTPUT_JSONL}'!")
    print("👉 Upload this JSONL file to Google AI Studio (aistudio.google.com) to fine-tune a custom Gemini model checkpoint.")

if __name__ == "__main__":
    export_jsonl()
