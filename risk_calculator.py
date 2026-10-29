"""
NapoVision Biomechanical Risk Index Calculator
Computes evidence-based ergonomic strain scores (0-100) and multi-zone anatomical stress assessments.
"""

def compute_biomechanical_risk(predicted_class, confidence, cam_mask=None):
    """
    Computes numerical risk index (0-100), severity level, and anatomical breakdown.
    
    Classes:
      - 'Supine/Back': Lowest strain (10-25)
      - 'Side': Mild-to-moderate strain (25-45)
      - 'Fetal': Moderate strain (50-68)
      - 'Prone/Stomach': High/Severe strain (75-92)
    """
    conf = float(confidence)
    
    # Base risk profiles & anatomical parameters
    if predicted_class == 'Prone/Stomach':
        base_score = 82
        jitter = int((conf - 0.5) * 10)
        risk_score = min(96, max(75, base_score + jitter))
        risk_level = "High Risk"
        risk_color = "#EF4444" # Rose
        status_text = "Severe Cervical Rotation & Lumbar Hyperextension"
        
        breakdown = {
            "cervical": min(98, int(risk_score * 1.12)),
            "thoracic": int(risk_score * 0.78),
            "lumbar": min(95, int(risk_score * 1.05)),
            "pelvic": int(risk_score * 0.82)
        }
        
        strain_zones = [
            {
                "zone": "Cervical Spine (C1-C7)",
                "status": "Extreme Rotational Torsion (~90°)",
                "severity": "Critical",
                "impact": "Suboccipital compression, facet pinching & vertebral artery stretch"
            },
            {
                "zone": "Lumbar Lordosis (L1-S1)",
                "status": "Hyper-lordotic Shearing",
                "severity": "High",
                "impact": "Excessive anterior pelvic tilt & posterior disc pinching"
            },
            {
                "zone": "Thoracic & Chest Wall",
                "status": "Anterior Compression",
                "severity": "Moderate",
                "impact": "Restricted rib cage expansion and diaphragmatic breathing"
            },
            {
                "zone": "Shoulder Girdle",
                "status": "Glenohumeral Impingement",
                "severity": "Moderate",
                "impact": "Rotator cuff compression from raised arm placement"
            }
        ]
        
    elif predicted_class == 'Fetal':
        base_score = 62
        jitter = int((conf - 0.5) * 8)
        risk_score = min(74, max(52, base_score + jitter))
        risk_level = "Moderate Risk"
        risk_color = "#F59E0B" # Amber
        status_text = "Excessive Spinal Hyperflexion & Hip Flexor Tightness"
        
        breakdown = {
            "cervical": int(risk_score * 0.92),
            "thoracic": min(88, int(risk_score * 1.15)),
            "lumbar": min(86, int(risk_score * 1.08)),
            "pelvic": int(risk_score * 0.95)
        }
        
        strain_zones = [
            {
                "zone": "Thoracolumbar Spine",
                "status": "Sustained C-Curve Hyperflexion",
                "severity": "High",
                "impact": "Posterior disc herniation vector & paraspinal ligament strain"
            },
            {
                "zone": "Hip Flexors & Psoas",
                "status": "Acute Hip Flexion Contracture",
                "severity": "Moderate",
                "impact": "Adaptive shortening of psoas causing morning lumbar tightness"
            },
            {
                "zone": "Cervical Spine (C3-C7)",
                "status": "Forward Head Flexion",
                "severity": "Moderate",
                "impact": "Excessive cranial flexion and chin-to-chest crowding"
            },
            {
                "zone": "Diaphragm Excursion",
                "status": "Abdominal Visceral Cramping",
                "severity": "Mild",
                "impact": "Restricted deep abdominal diaphragmatic breath cycles"
            }
        ]
        
    elif predicted_class == 'Side':
        base_score = 35
        jitter = int((conf - 0.5) * 6)
        risk_score = min(48, max(24, base_score + jitter))
        risk_level = "Mild Risk"
        risk_color = "#3B82F6" # Soft Blue / Indigo
        status_text = "Good Base Alignment with Lateral Pelvic Torsion Risks"
        
        breakdown = {
            "cervical": int(risk_score * 1.05),
            "thoracic": int(risk_score * 0.85),
            "lumbar": int(risk_score * 0.98),
            "pelvic": min(65, int(risk_score * 1.20))
        }
        
        strain_zones = [
            {
                "zone": "Pelvis & Sacroiliac Joint",
                "status": "Rotational Torsion",
                "severity": "Moderate",
                "impact": "Top knee dropping onto mattress twists lower lumbar spine"
            },
            {
                "zone": "Cervical Spine (Lateral Tilt)",
                "status": "Potential Lateral Deviation",
                "severity": "Mild",
                "impact": "Pillow height mismatch causes lateral neck curvature"
            },
            {
                "zone": "Dependent Shoulder Joint",
                "status": "Direct Pressure Load",
                "severity": "Moderate",
                "impact": "Acromial compression under torso weight without mattress contouring"
            },
            {
                "zone": "Spinal Neutrality",
                "status": "Optimal Horizontal Plane",
                "severity": "Low",
                "impact": "Promotes healthy glymphatic drainage and open airways"
            }
        ]
        
    else:  # Supine/Back
        base_score = 16
        jitter = int((conf - 0.5) * 6)
        risk_score = min(28, max(10, base_score + jitter))
        risk_level = "Low Risk"
        risk_color = "#10B981" # Emerald
        status_text = "Optimal Anatomical Alignment & Neutral Load Distribution"
        
        breakdown = {
            "cervical": int(risk_score * 0.95),
            "thoracic": int(risk_score * 0.80),
            "lumbar": int(risk_score * 1.15),
            "pelvic": int(risk_score * 0.75)
        }
        
        strain_zones = [
            {
                "zone": "Full Spinal Column",
                "status": "Neutral Hydrostatic Dispersion",
                "severity": "Low",
                "impact": "Even intervertebral disc pressure with no torsional twisting"
            },
            {
                "zone": "Lumbar Lordosis (L4-S1)",
                "status": "Mild Arch Gap",
                "severity": "Mild",
                "impact": "Risk of mild hyperextension sagging on overly firm mattresses"
            },
            {
                "zone": "Cervical Lordosis",
                "status": "Balanced Cranial Support",
                "severity": "Low",
                "impact": "Maintains 20-30° natural cervical curve with proper contoured pillow"
            },
            {
                "zone": "Upper Airway & Velopharynx",
                "status": "Gravitational Base-of-Tongue Fall",
                "severity": "Mild",
                "impact": "Potential snoring or airway narrowing if head is not elevated ~10°"
            }
        ]
        
    return {
        "score": int(risk_score),
        "level": risk_level,
        "color": risk_color,
        "status_text": status_text,
        "anatomical_breakdown": breakdown,
        "strain_zones": strain_zones
    }
