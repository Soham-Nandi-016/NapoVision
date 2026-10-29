import os
import io
import base64
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from PIL import Image
import torch

from model_utils import (
    CLASSES,
    CLASS_DISPLAY_NAMES,
    build_model,
    GradCAM,
    get_transforms,
    overlay_heatmap_on_image
)
from risk_calculator import compute_biomechanical_risk
from groq_engine import get_clinical_advice

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("napovision_app")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Global model state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "napovision_model.pth")

logger.info(f"Using compute device: {DEVICE}")

# Initialize model
model = build_model(num_classes=len(CLASSES))
if os.path.exists(MODEL_PATH):
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded NapoVision model weights from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint from {MODEL_PATH}: {e}")
else:
    logger.warning(f"No checkpoint found at {MODEL_PATH}. Initialized base MobileNetV2 architecture.")

model.to(DEVICE)
model.eval()

# Grad-CAM engine
gradcam_engine = GradCAM(model, target_layer=model.features[-1])
transform_pipeline = get_transforms()


@app.route("/")
def index():
    """Serves the main NapoVision web interface"""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns system status, device configuration, and model availability"""
    groq_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    return jsonify({
        "status": "online",
        "system": "NapoVision Medical XAI Engine",
        "tagline": "Sleep easy, live easy",
        "device": str(DEVICE),
        "model_loaded": os.path.exists(MODEL_PATH),
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
    Returns class prediction, confidence, Grad-CAM heatmap (base64),
    Biomechanical Risk Index (0-100), and Groq Clinical Advice.
    """
    try:
        raw_image_data = None
        user_groq_key = request.headers.get("X-Groq-Api-Key") or request.form.get("groq_api_key")

        # 1. Parse image input
        if "file" in request.files:
            file = request.files["file"]
            raw_image_data = file.read()
        elif "image" in request.files:
            file = request.files["image"]
            raw_image_data = file.read()
        elif request.is_json:
            data = request.get_json()
            if not user_groq_key and "groq_api_key" in data:
                user_groq_key = data["groq_api_key"]
            if "image" in data:
                img_str = data["image"]
                if "," in img_str:
                    img_str = img_str.split(",", 1)[1]
                raw_image_data = base64.b64decode(img_str)

        if not raw_image_data:
            return jsonify({"error": "No image payload found in request. Please upload a file or base64 data string."}), 400

        # 2. Open and preprocess image
        original_pil = Image.open(io.BytesIO(raw_image_data)).convert("RGB")
        input_tensor = transform_pipeline(original_pil).unsqueeze(0).to(DEVICE)

        # 3. Model Inference & Grad-CAM Generation
        cam_mask, predicted_class_idx, logits = gradcam_engine.generate(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        # All class probabilities mapping
        class_probs = {
            CLASSES[i]: float(probabilities[i])
            for i in range(len(CLASSES))
        }

        # 4. Generate Explainable AI (Grad-CAM) Visual Heatmaps
        blended_b64, heatmap_only_b64 = overlay_heatmap_on_image(original_pil, cam_mask, alpha=0.55)

        # 5. Compute Biomechanical Strain Risk Index
        risk_data = compute_biomechanical_risk(predicted_class, confidence, cam_mask)

        # 6. Groq LLM Clinical Ergonomics Synthesis
        clinical_advice = get_clinical_advice(
            posture_class=predicted_class,
            confidence=confidence,
            risk_data=risk_data,
            user_api_key=user_groq_key
        )

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
    logger.info(f"Starting NapoVision server at http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
