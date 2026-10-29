from PIL import Image
import torch
from model_utils import build_model, GradCAM, get_transforms, overlay_heatmap_on_image, CLASSES
from risk_calculator import compute_biomechanical_risk
from groq_engine import get_clinical_advice

model = build_model(4)
model.load_state_dict(torch.load('napovision_model.pth', map_location='cpu'))
model.eval()

gradcam = GradCAM(model)
transform = get_transforms()

for name, path in [('Supine', 'static/presets/supine.jpg'), ('Side', 'static/presets/side.jpg'), ('Fetal', 'static/presets/fetal.jpg'), ('Prone', 'static/presets/prone.jpg')]:
    img = Image.open(path).convert('RGB')
    tensor = transform(img).unsqueeze(0)
    cam, pred_idx, logits = gradcam.generate(tensor)
    probs = torch.softmax(logits, dim=1)[0]
    pred_cls = CLASSES[pred_idx]
    conf = float(probs[pred_idx])
    risk = compute_biomechanical_risk(pred_cls, conf, cam)
    blended_b64, heatmap_b64 = overlay_heatmap_on_image(img, cam)
    advice = get_clinical_advice(pred_cls, conf, risk)
    
    print(f"[{name}] Predicted: {pred_cls} | Conf: {conf*100:.2f}% | Risk: {risk['score']}/100 ({risk['level']}) | Advice Engine: {advice['engine_source']}")
    print(f"       GradCAM Overlay generated length: {len(blended_b64)} chars")
