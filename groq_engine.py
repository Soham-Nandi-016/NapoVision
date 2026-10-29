"""
NapoVision Groq Clinical Ergonomic Engine
Integrates Groq API to translate technical posture classification,
Grad-CAM saliency heatmaps, and biomechanical strain metrics into structured clinical prescriptions.
"""

import os
import json
import logging
from dotenv import load_dotenv

# Load environment configuration (.env)
load_dotenv()

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger("groq_engine")

CLINICAL_SYSTEM_PROMPT = """You are Dr. NapoVision, an elite Clinical Biomechanist, Orthopedic Physical Therapist, and Sleep Ergonomics Specialist with decades of clinical experience.
Your role is to translate computer vision sleep posture classifications, Biomechanical Strain Risk Index values (0-100), and Explainable AI (Grad-CAM) spinal strain saliency maps into empathetic, highly rigorous, evidence-based orthopedic prescriptions.

Respond strictly with valid JSON conforming to this schema:
{
  "clinical_summary": "A 2-3 sentence executive clinical assessment summarizing the patient's spinal alignment and risk tier.",
  "pathomechanics": "Detailed explanation of the anatomical vectors, facet joint compression, disc shearing, or muscular imbalances detected in this posture.",
  "cervical_prescription": "Exact orthopedic recommendations for the cervical spine (C1-C7), including pillow loft (cm/inches), neck curvature support, and head positioning.",
  "lumbar_prescription": "Targeted corrective measures for lumbar lordosis (L1-S1), pelvic tilt, and sacroiliac joint neutralization.",
  "ergonomic_equipment": "Specific sleep system recommendations (mattress firmness score 1-10, memory foam/latex density, ergonomic contour pillows, knee wedges).",
  "corrective_exercises": [
    {
      "name": "Name of clinical stretch or exercise",
      "dosage": "e.g., 2 sets of 30 seconds hold",
      "instruction": "Step-by-step biomechanical execution instructions."
    }
  ],
  "long_term_prognosis": "Clinical forecast of chronic conditions prevented (e.g., facet arthrosis, cervical radiculopathy, disc herniation, piriformis syndrome)."
}
"""

# Candidate models in prioritized order
CANDIDATE_MODELS = [
    os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.1-8b-instant"
]

def get_clinical_advice(posture_class, confidence, risk_data, user_api_key=None):
    """
    Generates dynamic clinical ergonomic advice using Groq API bound directly to
    the classified class, confidence, biomechanical risk score, and Grad-CAM strain zones.
    """
    api_key = user_api_key or os.environ.get("GROQ_API_KEY", "").strip()

    if GROQ_AVAILABLE and api_key:
        try:
            client = Groq(api_key=api_key)
            strain_zone_descriptions = [
                f"{z['zone']} - {z['status']} (Impact: {z['impact']})"
                for z in risk_data.get('strain_zones', [])
            ]
            
            prompt_content = f"""
PATIENT POSTURE DIAGNOSTIC REPORT:
- Classified Sleep Posture: {posture_class}
- Computer Vision Model Confidence: {confidence*100:.2f}%
- Biomechanical Strain Risk Index: {risk_data['score']}/100 ({risk_data['level']})
- Primary Stress Profile: {risk_data['status_text']}
- Anatomical Strain Scores:
  * Cervical Spine: {risk_data['anatomical_breakdown']['cervical']}/100
  * Thoracic Spine: {risk_data['anatomical_breakdown']['thoracic']}/100
  * Lumbar Spine: {risk_data['anatomical_breakdown']['lumbar']}/100
  * Pelvic/Hip Alignment: {risk_data['anatomical_breakdown']['pelvic']}/100
- Grad-CAM Strain Heatmap Observations:
  * Identified High-Stress Anatomical Zones: {'; '.join(strain_zone_descriptions)}

CLINICAL INSTRUCTIONS:
Generate a personalized, context-aware clinical ergonomics evaluation strictly conforming to the JSON schema.
- In 'clinical_summary', explicitly cite the diagnosed {posture_class} posture, the model confidence ({confidence*100:.2f}%), and the measured risk tier ({risk_data['level']} with score {risk_data['score']}/100).
- In 'pathomechanics', analyze the exact anatomical vectors, joint compression, and muscle strains specific to {posture_class}.
- In 'cervical_prescription' and 'lumbar_prescription', prescribe targeted biomechanical corrections addressing the patient's cervical strain ({risk_data['anatomical_breakdown']['cervical']}/100) and lumbar strain ({risk_data['anatomical_breakdown']['lumbar']}/100).
- In 'ergonomic_equipment', provide specific mattress firmness recommendations and pillow loft measurements tailored to {posture_class}.
- In 'corrective_exercises', deliver 3 to 4 clinical rehabilitation exercises specifically designed to counter the muscular imbalances of {posture_class}.
"""
            tried_models = []
            last_err = None
            
            for model_candidate in CANDIDATE_MODELS:
                if not model_candidate or model_candidate in tried_models:
                    continue
                tried_models.append(model_candidate)
                
                try:
                    logger.info(f"Invoking Groq clinical generation with model: {model_candidate} for class: {posture_class}")
                    completion = client.chat.completions.create(
                        model=model_candidate,
                        messages=[
                            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt_content}
                        ],
                        temperature=0.3,
                        max_tokens=1500,
                        response_format={"type": "json_object"}
                    )
                    raw_response = completion.choices[0].message.content
                    parsed = json.loads(raw_response)
                    parsed["engine_source"] = f"Groq {model_candidate} (Live Clinical Synthesis)"
                    logger.info(f"Groq generation successful with model: {model_candidate}")
                    return parsed
                except Exception as model_err:
                    last_err = model_err
                    err_msg = str(model_err)
                    if "model_not_found" in err_msg or "does not exist" in err_msg:
                        logger.warning(f"Model {model_candidate} not available on this Groq account, trying next...")
                        continue
                    else:
                        logger.warning(f"Groq API error with {model_candidate}: {model_err}")
                        break

            logger.warning(f"All Groq candidates exhausted. Last error: {last_err}. Using NapoVision Clinical Engine.")
            fallback = get_clinical_fallback(posture_class, risk_data)
            fallback["engine_source"] = f"NapoVision Clinical Engine (Fallback: {str(last_err)})"
            return fallback

        except Exception as e:
            logger.warning(f"Groq API client initialization error: {e}. Falling back to NapoVision Clinical Engine.")
            fallback = get_clinical_fallback(posture_class, risk_data)
            fallback["engine_source"] = f"NapoVision Clinical Engine (Fallback: {str(e)})"
            return fallback

    # If no API key provided, use rich clinical fallback engine
    fallback = get_clinical_fallback(posture_class, risk_data)
    fallback["engine_source"] = "NapoVision Clinical Ergonomic Engine (Enter Groq API Key for Dynamic Synthesis)"
    return fallback


def get_clinical_fallback(posture_class, risk_data):
    """
    Evidence-based clinical fallback profiles formulated by orthopedic physical therapy guidelines.
    """
    score = risk_data['score']

    if posture_class == 'Prone/Stomach':
        return {
            "clinical_summary": f"Diagnostic evaluation confirms a Prone (Stomach Sleeping) posture with an elevated Biomechanical Strain Index of {score}/100 (High Risk). This posture represents the most biomechanically hazardous sleeping position due to sustained cervical rotational torque and forced lumbar hyperextension.",
            "pathomechanics": "To maintain a patent airway in the prone position, the cervical spine must undergo an extreme rotational torque of approximately 80° to 90° for prolonged hours. This causes continuous unilateral compression of the facet joints (C2-C6), tension strain on the contralateral levator scapulae and sternocleidomastoid, and shear stress across the vertebral arteries. Simultaneously, gravity pulls the abdominal contents downward, accentuating lumbar lordosis and creating compressive pinching in the posterior elements of L4-S1.",
            "cervical_prescription": "Immediately eliminate thick pillows under the head, which exacerbate cervical hyperextension. If transitioning away from prone sleeping is challenging, utilize an ultra-low profile contoured pillow (under 4 cm / 1.5 inches) or place a soft forehead cradle to minimize rotational torque on C1-C7.",
            "lumbar_prescription": "Place a thin, flat orthopedic pillow or folded towel directly beneath the lower abdomen and pelvis (anterior superior iliac spines). This elevates the pelvic basin, neutralizes hyper-lordotic lumbar sagging, and decompresses posterior disc margins.",
            "ergonomic_equipment": "Transition to a Medium-Firm to Firm mattress (7.0-8.0 out of 10 on the firmness scale) with high-density resilient base foam to prevent pelvic sinking. Consider a body positioning bolster to assist gradual transition toward lateral or semi-recumbent posture.",
            "corrective_exercises": [
                {
                    "name": "Prone Press-Up & Cat-Camel Decompression",
                    "dosage": "2 sets of 10 repetitions, smooth rhythmic cadence",
                    "instruction": "Perform gentle Cat-Camel mobilization on all fours to mobilize the thoracolumbar junction, followed by slow prone elbow props to restore passive extension balance."
                },
                {
                    "name": "Suboccipital Chin Tucks & Levator Scapulae Stretch",
                    "dosage": "Hold 20 seconds, repeat 3 times per side",
                    "instruction": "Gently retract the chin straight back creating a double chin, then tilt the head diagonally downward toward the opposite armpit to relieve suboccipital torsion."
                },
                {
                    "name": "Prone Hip Flexor / Rectus Femoris Release",
                    "dosage": "Hold 30 seconds per side",
                    "instruction": "Lie supine or standing, draw one heel toward the gluteus keeping knees aligned to decompress anterior hip tighteners."
                }
            ],
            "long_term_prognosis": "Transitioning out of unassisted prone sleeping prevents early-onset cervical facet arthrosis, disc bulge progression at L5-S1, morning occipital headaches, and chronic myofascial neck spasm."
        }

    elif posture_class == 'Fetal':
        return {
            "clinical_summary": f"Biomechanical assessment identifies a curled Fetal Posture with a Strain Index of {score}/100 (Moderate Risk). While comfortable and offering psychological security, the prolonged C-curve creates sustained posterior spinal tension and adaptive hip flexor shortening.",
            "pathomechanics": "The fetal position forces the entire axial skeleton into a uniform, acute flexion pattern. The lumbar lordosis is reversed into kyphosis, which increases intradiscal hydrostatic pressure anteriorly and shifts the nucleus pulposus backward against the posterior longitudinal ligament. Additionally, prolonged 90°+ hip flexion induces adaptive shortening of the psoas major and rectus femoris, leading to lower back stiffness upon rising.",
            "cervical_prescription": "Avoid excessive chin-to-chest tucking. Adjust pillow height so that the head rests in neutral horizontal alignment with the thoracic spine (pillow loft approximately 10-12 cm), keeping the cervical lordosis relaxed without acute forward flexion.",
            "lumbar_prescription": "Relax the acute hip and knee tuck. Open the angle between torso and thighs to greater than 100 degrees. Place an hourglass memory foam knee spacer between the knees to stabilize the pelvis and prevent sacroiliac joint torque.",
            "ergonomic_equipment": "A Medium-Soft to Medium mattress (5.5-6.5 on firmness scale) that contours around the lateral iliac crest and shoulder acromion without sagging. A full-length cylindrical body pillow encourages gentle spinal elongation.",
            "corrective_exercises": [
                {
                    "name": "Prone Sphinx Extension (McKenzie Protocol)",
                    "dosage": "Hold 45 seconds, 2 repetitions",
                    "instruction": "Lie prone on elbows, allowing the abdomen to sag into the mattress, gently introducing natural lumbar lordosis to counterbalance nighttime hyperflexion."
                },
                {
                    "name": "Half-Kneeling Psoas & Hip Flexor Stretch",
                    "dosage": "Hold 30 seconds each side, 2 sets",
                    "instruction": "Step forward into a lunge with the back knee on a cushion. Gently tuck the tailbone under (posterior pelvic tilt) and glide forward until an anterior hip stretch is felt."
                },
                {
                    "name": "Thoracic Open-Book Rotations",
                    "dosage": "8 repetitions per side",
                    "instruction": "Lie on side with knees bent, rotate the top arm and rib cage open toward the opposite floor to restore rotational mobility in the mid-back."
                }
            ],
            "long_term_prognosis": "Opening the fetal posture reduces risks of posterior disc protrusion, nocturnal thoracic outlet impingement, and chronic anterior hip tightness."
        }

    elif posture_class == 'Side':
        return {
            "clinical_summary": f"Patient demonstrates Lateral Side Recumbency with a Biomechanical Strain Index of {score}/100 (Mild Risk). Side sleeping is widely considered an ergonomically sound position, though it requires specific joint gap bridging to prevent pelvic rotation and shoulder impingement.",
            "pathomechanics": "In lateral recumbency, the spinal column should ideally form a straight horizontal line parallel to the mattress. If the pillow is too thin or too thick, lateral cervical flexion occurs. Furthermore, when the top leg drops forward onto the sleeping surface, it exerts rotational torque on the pelvis, creating torsional shearing forces at the L4-L5 and L5-S1 motion segments and stressing the sacroiliac ligamentous complex.",
            "cervical_prescription": "Measure the distance from the lateral tip of the acromion process to the earlobe. Select a contoured cervical pillow whose compressed loft exactly matches this distance (typically 10-14 cm), ensuring the cervical vertebrae remain in a straight horizontal vector.",
            "lumbar_prescription": "Place an ergonomically contoured knee spacer pillow (density 45-50 kg/m³) directly between the medial knees and ankles. This maintains parallel femur alignment, prevents internal hip rotation, and neutralizes pelvic torsion.",
            "ergonomic_equipment": "Medium to Medium-Firm mattress (6.0-7.0 firmness) with dedicated zoned shoulder pressure-relief zones. A contoured memory foam pillow with an ear indentation alleviates auricular chondritis and acromial pressure.",
            "corrective_exercises": [
                {
                    "name": "Side-Lying Clamshells with Pelvic Neutralizer",
                    "dosage": "2 sets of 12 reps per side",
                    "instruction": "Lie on your side with hips at 45° and knees at 90°. Keeping heels together, rotate top knee upward without letting the pelvis rock backward to activate the gluteus medius."
                },
                {
                    "name": "Pectoralis & Anterior Shoulder Doorway Stretch",
                    "dosage": "Hold 30 seconds, 2 repetitions",
                    "instruction": "Place forearm against a door jamb at 90° angle, step gently forward to decompress anterior shoulder capsule after hours of lateral compression."
                },
                {
                    "name": "Piriformis & Sciatic Nerve Gliding",
                    "dosage": "Hold 30 seconds per leg",
                    "instruction": "Perform figure-four stretch on your back, pulling opposite knee toward chest to release deep pelvic rotators."
                }
            ],
            "long_term_prognosis": "Properly supported side sleeping optimizes neuro-glymphatic waste clearance, maximizes cardiac hemodynamics, and eliminates asymmetric disc pressure."
        }

    else:  # Supine/Back
        return {
            "clinical_summary": f"Biomechanical analysis identifies an Optimal Supine (Back Sleeper) posture with a low Biomechanical Strain Index of {score}/100 (Low Risk / Healthy Alignment). This posture offers the most balanced hydrostatic pressure distribution across the intervertebral discs.",
            "pathomechanics": "Supine sleeping evenly distributes body weight across the broad dorsal surfaces (occiput, scapulae, sacrum, calves), minimizing localized tissue ischemia and joint shear. The primary biomechanical considerations are maintaining natural cervical lordosis without forward head thrust, and preventing excessive lumbar arch hyperextension that can occur when lying on an unyielding, overly stiff surface.",
            "cervical_prescription": "Use a medium-low cervical contour pillow (loft 7-9 cm) featuring a central cranial depression and an ergonomic neck roll support. The cervical lordotic curve should be supported while keeping the chin parallel to the chest (avoiding upward hyperextension or downward chin tuck).",
            "lumbar_prescription": "Place a low-profile cylindrical bolster or small pillow under the popliteal fossa (beneath the knees). Flexing the knees by 10-15° relaxes the iliopsoas muscle and allows the lumbar spine to settle comfortably into the mattress with zero gravitational shear.",
            "ergonomic_equipment": "Medium-Firm mattress (6.5-7.5 firmness) with adaptive contouring comfort layers (memory foam or natural latex) supported by pocketed orthopedic coils. Elevating the head of the bed 5-10° may be beneficial if snoring or mild upper-airway collapsibility occurs.",
            "corrective_exercises": [
                {
                    "name": "Pelvic Tilts & Gluteal Bridging",
                    "dosage": "2 sets of 12 repetitions",
                    "instruction": "Lie on back with knees bent, flatten the lower back against the floor by engaging abdominals, then gently lift hips to strengthen posterior chain stabilizing musculature."
                },
                {
                    "name": "Supine Knee-to-Chest Lumbar Release",
                    "dosage": "Hold 30 seconds each side",
                    "instruction": "Draw one knee toward your chest while keeping the opposite leg relaxed to gently lengthen lumbar paraspinals."
                },
                {
                    "name": "Chin Tucks in Neutral Supine",
                    "dosage": "10 repetitions with 3-second isometric holds",
                    "instruction": "Press the back of the neck gently toward the mattress while nodding head slightly downward to strengthen deep cervical neck flexors (longus colli)."
                }
            ],
            "long_term_prognosis": "Maintaining optimal supine ergonomics preserves natural spinal lordosis, prevents facial sleep wrinkles, and ensures minimal nocturnal joint inflammation."
        }
