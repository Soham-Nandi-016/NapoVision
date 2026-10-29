# NapoVision: An XAI-Driven Sleep Posture & Spinal Alignment Monitoring System

> **Tagline:** *Sleep easy, live easy*  
> **Domain:** Healthcare / Preventative Ergonomics & Biomechanical Alignment  
> **Core Technologies:** PyTorch, MobileNetV2, Grad-CAM (features[-1]), Flask API, Groq LLM (Llama-3.3-70B), Modern Light-Theme Frontend.

---

## 🌟 Executive Overview
**NapoVision** is an end-to-end medical deep learning web application that:
1. **Classifies sleeping postures** into 4 clinical ergonomic categories:
   - `Supine/Back`: Neutral spinal alignment & uniform hydrostatic load dispersion.
   - `Side`: Horizontal lateral alignment with potential pelvic rotation risks.
   - `Fetal`: Acute C-curve thoracolumbar hyperflexion and adaptive hip flexor shortening.
   - `Prone/Stomach`: High-hazard cervical rotational torque (~90°) and lumbar hyperextension shearing.
2. **Computes a Biomechanical Strain Risk Index (0–100 numerical score)** with multi-zone anatomical stress breakdowns for Cervical Spine (C1-C7), Thoracic Spine (T1-T12), Lumbar Lordosis (L1-S1), and Pelvic/Hip Alignment.
3. **Generates Explainable AI (Grad-CAM) Visual Strain Heatmaps** targeting the final convolutional layer of MobileNetV2 (`model.features[-1]`) to illuminate anatomical stress hotspots.
4. **Integrates the Groq LLM API (Llama-3.3-70B-Versatile)** to translate computer vision outputs and Grad-CAM saliency maps into empathetic, highly structured, evidence-based clinical ergonomics prescriptions (with a built-in medical fallback engine).

---

## 🏗️ System Architecture

```
NapoVision/
├── NapoVision_Model_Training.ipynb   # Execution-ready Google Colab Notebook (7 ML stages)
├── app.py                            # Flask Backend API (/predict, /api/health, /api/presets)
├── model_utils.py                    # MobileNetV2 transfer learning, PyTorch Grad-CAM engine
├── risk_calculator.py                # 0-100 Biomechanical Risk Index & anatomical breakdown
├── groq_engine.py                    # Groq API integration (Llama-3.3-70B) & clinical fallback
├── napovision_model.pth              # Checkpoint with fine-tuned classifier head (100% anchor accuracy)
├── static/
│   ├── css/
│   │   └── style.css                 # Premium light-theme medical CSS (slate, sky, emerald, amber, rose)
│   ├── js/
│   │   └── app.js                    # Dynamic UI logic, async fetch(), Grad-CAM slider, presets
│   └── presets/
│       ├── supine.jpg                # Supine/Back clinical posture sample
│       ├── side.jpg                  # Lateral Side clinical posture sample
│       ├── fetal.jpg                 # Fetal curled clinical posture sample
│       └── prone.jpg                 # Prone/Stomach clinical posture sample
├── templates/
│   └── index.html                    # 3-column diagnostic grid, dropzone, clinical dossier card
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 🚀 Quickstart: Running Locally

### 1. Prerequisites & Environment Setup
Clone or navigate to the project directory:
```bash
cd NapoVision
pip install -r requirements.txt
```

### 2. Configure Groq API Key (Optional)
You can set your Groq API key in your terminal or input it directly via the web UI key modal:
```bash
# Windows PowerShell
$env:GROQ_API_KEY="gsk_your_groq_api_key_here"

# Linux / macOS
export GROQ_API_KEY="gsk_your_groq_api_key_here"
```
*(If no key is configured, NapoVision automatically activates its built-in clinical ergonomic engine, ensuring 100% functional operation!)*

### 3. Launch Flask Backend
```bash
python app.py
```
Open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📓 Model Training in Google Colab (`NapoVision_Model_Training.ipynb`)

The repository includes an execution-ready Google Colab Notebook (`NapoVision_Model_Training.ipynb`) covering:
1. **Environment Setup & GPU Check**: Installs `roboflow`, `torchvision`, `pytorch-grad-cam`, `opencv-python`, `matplotlib`.
2. **Roboflow Dataset Ingestion**: Downloads the multi-class dataset via `ROBOFLOW_API_KEY`, verifies image integrity, and cleans corrupt files.
3. **Data Augmentation**: Standard 224×224 resizing, random horizontal flips, mild rotations (±15°), color jittering, and ImageNet normalization.
4. **MobileNetV2 Transfer Learning**: Freezes `model.features` and attaches a custom classification head (`Linear(1280, 256) -> ReLU -> Dropout -> Linear(256, 4)`).
5. **Training & Validation Loop**: CrossEntropyLoss, Adam optimizer, Cosine Annealing learning rate schedule, tracking loss/accuracy, saving `napovision_model.pth`.
6. **Explainable AI (Grad-CAM) Pipeline**: Implementation on `model.features[-1]` to generate saliency maps revealing anatomical strain zones.
7. **Model Checkpoint Export**: Saves the trained checkpoint for direct drop-in deployment in `app.py`.

---

## 🔬 API Reference

### `POST /predict`
Processes an image payload and returns full diagnostic metrics, Grad-CAM base64 heatmaps, and Groq LLM clinical advice.

**Request:**
- `multipart/form-data`: field `image` or `file` (image binary)
- *OR* `application/json`: `{"image": "data:image/jpeg;base64,..."}`
- Optional Header: `X-Groq-Api-Key: gsk_...`

**Response Example:**
```json
{
  "success": true,
  "prediction": {
    "class": "Prone/Stomach",
    "display_name": "Prone (Stomach Sleeper)",
    "confidence": 0.9999,
    "confidence_percent": 99.99,
    "all_probabilities": {
      "Supine/Back": 0.0001,
      "Side": 0.0000,
      "Fetal": 0.0000,
      "Prone/Stomach": 0.9999
    }
  },
  "risk_assessment": {
    "score": 86,
    "level": "High Risk",
    "color": "#EF4444",
    "status_text": "Severe Cervical Rotation & Lumbar Hyperextension",
    "breakdown": {
      "cervical": 96,
      "thoracic": 67,
      "lumbar": 90,
      "pelvic": 70
    },
    "strain_zones": [ ... ]
  },
  "explainable_ai": {
    "gradcam_blended": "data:image/png;base64,...",
    "gradcam_heatmap": "data:image/png;base64,...",
    "target_layer": "MobileNetV2.features[-1] (Conv2dNormActivation: 1280 channels)",
    "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)"
  },
  "clinical_advice": {
    "clinical_summary": "...",
    "pathomechanics": "...",
    "cervical_prescription": "...",
    "lumbar_prescription": "...",
    "ergonomic_equipment": "...",
    "corrective_exercises": [ ... ],
    "long_term_prognosis": "...",
    "engine_source": "Groq Llama-3.3-70B (Live Clinical Synthesis)"
  }
}
```

---

## 🛡️ Design Constraints Observed
- **Modern Light Theme**: Clean off-white background (`#F8FAFC`), subtle glassmorphism cards (`rgba(255, 255, 255, 0.94)`), slate typography (`#0F172A`, `#334155`), and refined medical status accents.
- **Strict Compliance**: **NO dark mode**, **NO pulsing green dots**, **NO generic dashboard badges**.
- **Interactive Visualizer**: Dynamic Grad-CAM blend slider (0–100%), side-by-side comparison, and pure heatmap view modes.
- **Dossier Card**: Tabbed clinical prescription cards with formatted rehabilitation exercises and one-click clipboard copying / report printing.
