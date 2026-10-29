import io
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
import cv2

CLASSES = ['Supine/Back', 'Side', 'Fetal', 'Prone/Stomach']
CLASS_DISPLAY_NAMES = {
    'Supine/Back': 'Supine (Back Sleeper)',
    'Side': 'Lateral Side Sleeper',
    'Fetal': 'Fetal (Curled Posture)',
    'Prone/Stomach': 'Prone (Stomach Sleeper)'
}

def build_model(num_classes=4):
    """
    Constructs MobileNetV2 transfer learning architecture with custom linear classifier head.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # Freeze feature backbone
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Custom classification head
    in_features = model.last_channel
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

class GradCAM:
    """
    PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
    targeting the final convolutional feature layer of MobileNetV2 (features[-1]).
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer if target_layer is not None else model.features[-1]
        
        # Ensure target layer parameters track gradients for Grad-CAM backprop
        for param in self.target_layer.parameters():
            param.requires_grad = True

        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Computes Grad-CAM saliency map for the specified class (or top predicted class).
        input_tensor shape: (1, 3, H, W)
        """
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        score = output[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0] # (C, H, W)
        activations = self.activations[0] # (C, H, W)

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True) # (C, 1, 1)
        cam = torch.sum(weights * activations, dim=0).detach().cpu().numpy() # (H, W)

        # Apply ReLU to retain features that have positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, output

def get_transforms():
    """Standard ImageNet preprocessing transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def overlay_heatmap_on_image(original_pil_image, cam_mask, alpha=0.55):
    """
    Overlays 2D Grad-CAM saliency map (0 to 1) onto original PIL Image using OpenCV COLORMAP_JET.
    Returns:
      blended_base64: 'data:image/png;base64,...'
      heatmap_only_base64: 'data:image/png;base64,...'
    """
    orig_np = np.array(original_pil_image)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_RGBA2RGB)

    orig_h, orig_w = orig_np.shape[:2]

    # Resize CAM to match original image dimensions
    cam_resized = cv2.resize(cam_mask, (orig_w, orig_h))
    cam_uint8 = np.uint8(255 * cam_resized)

    # Generate COLORMAP_JET
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    blended = cv2.addWeighted(orig_np, float(alpha), heatmap_rgb, float(1.0 - alpha), 0)

    # Convert to base64 PNGs
    def to_b64(img_array):
        pil_img = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')

    blended_b64 = to_b64(blended)
    heatmap_only_b64 = to_b64(heatmap_rgb)

    return blended_b64, heatmap_only_b64
