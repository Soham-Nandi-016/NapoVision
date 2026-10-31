import os
import io
import base64
import logging
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np

# Load environment configuration (.env)
load_dotenv()

from model_utils import (
    CLASSES,
    CLASS_DISPLAY_NAMES,
    load_trained_model,
    data_transforms,
    get_transforms,
    GradCAM,
    overlay_heatmap_on_image
)
from risk_calculator import compute_biomechanical_risk
from groq_engine import get_clinical_advice

# Configure detailed application logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("napovision_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global model state & device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"NapoVision compute device: {DEVICE}")

# Initialize model using automated discovery & checkpoint verification
model, MODEL_PATH = load_trained_model(device=DEVICE)

# Grad-CAM engine attached to final convolutional feature layer of MobileNetV2
gradcam_engine = GradCAM(model, target_layer=model.features[-1])


@app.route("/")
def index():
    """Serves the main NapoVision web interface"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns system status, device configuration, model availability, and Groq status"""
    groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    model_exists = os.path.exists(MODEL_PATH) if MODEL_PATH else False
    return jsonify({
        "status": "online",
        "system": "NapoVision Medical XAI Engine",
        "tagline": "Sleep easy, live easy",
        "device": str(DEVICE),
        "model_loaded": model_exists,
        "model_path": MODEL_PATH,
        "groq_api_configured": groq_key_set,
        "supported_classes": CLASSES
    })


@app.route("/api/presets", methods=["GET"])
def get_presets():
    """Returns curated preset sleep posture sample references for quick testing"""
    presets = [
        {
            "id": "supine",
            "name": "Supine (Back)",
            "description": "Neutral spinal distribution with minimal joint stress",
            "image_url": "/static/presets/supine.jpg",
            "expected_risk": "Low (10-25)"
        },
        {
            "id": "side",
            "name": "Lateral Side",
            "description": "Horizontal spine alignment with potential pelvic torsion",
            "image_url": "/static/presets/side.jpg",
            "expected_risk": "Mild (25-45)"
        },
        {
            "id": "fetal",
            "name": "Fetal Curled",
            "description": "Continuous thoracolumbar C-curve hyperflexion",
            "image_url": "/static/presets/fetal.jpg",
            "expected_risk": "Moderate (50-68)"
        },
        {
            "id": "prone",
            "name": "Prone (Stomach)",
            "description": "Severe cervical rotation & lumbar hyperextension",
            "image_url": "/static/presets/prone.jpg",
            "expected_risk": "High (75-92)"
        }
    ]
    return jsonify({"presets": presets})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main diagnostic endpoint:
    Accepts image via multipart form-data ('image' / 'file') or JSON ('image' as base64 data URL).
    Decodes image bytes, executes live PyTorch MobileNetV2 forward pass via data_transforms['val'],
    computes dynamic Grad-CAM saliency map on target layer, calculates Biomechanical Risk,
    and invokes Groq LLM API with full diagnostic payload.
    """
    try:
        raw_image_data = None
        filename = "unknown_input.jpg"
        user_groq_key = request.headers.get("X-Groq-Api-Key") or request.form.get("groq_api_key")

        # 1. Parse Image Payload (File Upload, Base64, or Preset Reference)
        if "file" in request.files and request.files["file"].filename:
            file = request.files["file"]
            raw_image_data = file.read()
            filename = file.filename
        elif "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            raw_image_data = file.read()
            filename = file.filename
        elif request.form.get("preset") or request.form.get("preset_id"):
            preset_name = request.form.get("preset") or request.form.get("preset_id")
            preset_path = os.path.join(os.path.dirname(__file__), "static", "presets", f"{preset_name}.jpg")
            if os.path.exists(preset_path):
                with open(preset_path, "rb") as pf:
                    raw_image_data = pf.read()
                filename = f"{preset_name}.jpg"
        elif request.is_json:
            data = request.get_json()
            if not user_groq_key and "groq_api_key" in data:
                user_groq_key = data["groq_api_key"]

            if "preset" in data:
                preset_name = data["preset"]
                preset_path = os.path.join(os.path.dirname(__file__), "static", "presets", f"{preset_name}.jpg")
                if os.path.exists(preset_path):
                    with open(preset_path, "rb") as pf:
                        raw_image_data = pf.read()
                    filename = f"{preset_name}.jpg"
            elif "image" in data:
                img_str = data["image"]
                filename = data.get("filename", "base64_upload.jpg")
                if "," in img_str:
                    img_str = img_str.split(",", 1)[1]
                raw_image_data = base64.b64decode(img_str)

        if not raw_image_data:
            return jsonify({"error": "No image payload found in request. Please upload an image file, preset name, or base64 data string."}), 400

        # 2. Decode Raw Image Bytes to RGB PIL Image
        original_pil = Image.open(io.BytesIO(raw_image_data)).convert("RGB")
        orig_w, orig_h = original_pil.size

        # 3. Apply Validation Data Transforms & Transfer to Device
        val_transform = data_transforms["val"]
        input_tensor = val_transform(original_pil).unsqueeze(0).to(DEVICE)

        # 4. Live PyTorch Inference Forward Pass & Dynamic Grad-CAM Generation
        # Fresh forward pass on the active input_tensor:
        cam_mask, predicted_class_idx, logits = gradcam_engine.generate(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        # Compute full multi-class probability breakdown
        class_probs = {
            CLASSES[i]: float(probabilities[i])
            for i in range(len(CLASSES))
        }

        # 5. Explainable AI: Generate Grad-CAM Heatmap overlay accurately scaled to (orig_h, orig_w)
        blended_b64, heatmap_only_b64 = overlay_heatmap_on_image(original_pil, cam_mask, alpha=0.55)

        # 6. Biomechanical Strain Risk Index Calculation
        risk_data = compute_biomechanical_risk(predicted_class, confidence, cam_mask)

        # 7. Groq LLM Clinical Ergonomics Prescription
        clinical_advice = get_clinical_advice(
            posture_class=predicted_class,
            confidence=confidence,
            risk_data=risk_data,
            user_api_key=user_groq_key
        )

        # 8. Terminal Request Logging
        logger.info(
            f"[INFERENCE] Input: '{filename}' ({orig_w}x{orig_h}) | "
            f"Predicted Class: '{predicted_class}' ({CLASS_DISPLAY_NAMES.get(predicted_class)}) | "
            f"Confidence: {confidence * 100:.2f}% | "
            f"Biomechanical Risk: {risk_data['score']}/100 ({risk_data['level']}) | "
            f"Groq Engine: '{clinical_advice.get('engine_source')}'"
        )

        # 9. Format API Response Payload
        response_payload = {
            "success": True,
            "prediction": {
                "class": predicted_class,
                "display_name": CLASS_DISPLAY_NAMES.get(predicted_class, predicted_class),
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2),
                "all_probabilities": class_probs
            },
            "risk_assessment": {
                "score": risk_data["score"],
                "level": risk_data["level"],
                "color": risk_data["color"],
                "status_text": risk_data["status_text"],
                "breakdown": risk_data["anatomical_breakdown"],
                "strain_zones": risk_data["strain_zones"]
            },
            "explainable_ai": {
                "gradcam_blended": blended_b64,
                "gradcam_heatmap": heatmap_only_b64,
                "target_layer": "MobileNetV2.features[-1] (Conv2dNormActivation: 1280 channels)",
                "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)"
            },
            "clinical_advice": clinical_advice
        }

        return jsonify(response_payload)

    except Exception as e:
        logger.exception("Error processing prediction request")
        return jsonify({"error": f"Diagnostic processing failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting NapoVision server on host 0.0.0.0, port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
