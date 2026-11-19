"""Plain-language crop-care guidance for each PlantVillage class.

General integrated-pest-management advice only. Products and rates depend on country, crop stage
and the product label - always confirm with a local agricultural extension officer.
"""
from __future__ import annotations

import re

# kind: fungal | bacterial | viral | pest | healthy ; urgency: low | medium | high
D = {
    "Apple___Apple_scab": ("fungal", "medium", "Fungus (Venturia inaequalis). Olive-green to brown, velvety spots on leaves and fruit; severe cases cause early leaf drop.",
        ["Rake up and destroy fallen leaves - they carry the fungus over winter.", "Prune to open the canopy so leaves dry faster.", "Use a protectant fungicide from green-tip to petal-fall as per label and local advice.", "Plant scab-resistant varieties when replanting."]),
    "Apple___Black_rot": ("fungal", "medium", "Fungus (Botryosphaeria obtusa). 'Frogeye' leaf spots, rotting fruit and cankers on limbs.",
        ["Prune out dead wood and cankers and burn or bury them.", "Remove mummified fruit from the tree and ground.", "Apply an early-season protectant fungicide following the label.", "Keep trees vigorous - avoid drought and wounding stress."]),
    "Apple___Cedar_apple_rust": ("fungal", "medium", "Fungus (Gymnosporangium juniperi-virginianae) that alternates between apple and juniper/cedar. Bright yellow-orange leaf spots.",
        ["Remove nearby juniper/cedar galls if practical.", "Apply fungicide from pink bud through petal fall if pressure is high.", "Choose rust-resistant apple varieties.", "Rake and destroy badly infected leaves."]),
    "Cherry_(including_sour)___Powdery_mildew": ("fungal", "medium", "Fungus (Podosphaera clandestina). White powdery growth on young leaves and shoots, leaves may curl.",
        ["Prune for airflow and light penetration.", "Apply sulfur or another labelled fungicide at first sign, especially on new growth.", "Avoid heavy nitrogen that pushes soft new growth.", "Water at the base, not over the canopy."]),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("fungal", "medium", "Fungus (Cercospora zeae-maydis). Narrow rectangular tan-to-gray lesions running between leaf veins.",
        ["Rotate away from maize for at least a year.", "Bury or break down crop residue after harvest.", "Plant tolerant hybrids.", "Consider a foliar fungicide around tasseling if the disease is on upper leaves in warm, humid weather."]),
    "Corn_(maize)___Common_rust_": ("fungal", "low", "Fungus (Puccinia sorghi). Small cinnamon-brown pustules on both leaf surfaces.",
        ["Usually minor on resistant hybrids - monitor before spending on sprays.", "Plant resistant hybrids next season.", "Treat with a labelled fungicide only if pustules appear early and spread fast.", "Avoid very late planting."]),
    "Corn_(maize)___Northern_Leaf_Blight": ("fungal", "medium", "Fungus (Exserohilum turcicum). Long, cigar-shaped gray-green to tan lesions.",
        ["Rotate crops and manage residue.", "Use resistant hybrids.", "Scout from knee-high stage; a foliar fungicide can pay off if lesions reach the ear leaf before silking.", "Balance fertility - avoid excess nitrogen."]),
    "Grape___Black_rot": ("fungal", "high", "Fungus (Guignardia bidwellii). Brown circular leaf spots with tiny black dots, and berries that shrivel into hard black mummies.",
        ["Remove mummified berries and infected canes during pruning.", "Open the canopy (shoot thinning, leaf removal near clusters).", "Start protectant fungicide sprays at early shoot growth through a few weeks after bloom.", "Act early - young berries are the most vulnerable."]),
    "Grape___Esca_(Black_Measles)": ("fungal", "high", "Complex of wood-rotting fungi. 'Tiger-stripe' yellow/brown leaf patterns and dark spotting on berries; vines can die suddenly.",
        ["There is no cure - mark affected vines.", "Prune in dry weather and seal large pruning wounds.", "Cut out or remove severely affected vines and burn the wood.", "Avoid big pruning cuts and water stress."]),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("fungal", "medium", "Fungal leaf spot (Pseudocercospora vitis). Dark brown angular spots, mostly on older leaves, can cause early defoliation.",
        ["Remove and destroy fallen infected leaves.", "Improve airflow through canopy management.", "Apply a labelled fungicide if spots spread to younger leaves.", "Avoid overhead irrigation."]),
    "Orange___Haunglongbing_(Citrus_greening)": ("bacterial", "high", "Huanglongbing (HLB, citrus greening) - bacterium spread by the Asian citrus psyllid. Blotchy yellow mottling, lopsided bitter fruit, tree decline.",
        ["Report suspected HLB to your local agriculture authority - it is a regulated disease in many countries.", "There is no cure: remove confirmed infected trees promptly so they do not feed psyllids.", "Control the psyllid vector with a regional spray programme and scout new flushes.", "Plant only certified disease-free nursery trees."]),
    "Peach___Bacterial_spot": ("bacterial", "medium", "Bacterium (Xanthomonas arboricola pv. pruni). Small angular dark spots that drop out leaving 'shot-holes'; fruit pitting and cracking.",
        ["Choose tolerant varieties for new plantings.", "Copper-based bactericide sprays at leaf fall and early season (follow label - copper can burn leaves).", "Avoid overhead irrigation and excess nitrogen.", "Keep trees well watered and nourished to reduce stress."]),
    "Pepper,_bell___Bacterial_spot": ("bacterial", "medium", "Bacterium (Xanthomonas). Water-soaked spots turning brown on leaves, raised scabby spots on fruit.",
        ["Start with disease-free seed and transplants.", "Spray copper-based products as labelled when conditions are wet and warm.", "Rotate away from peppers and tomatoes for 2-3 years.", "Do not work among wet plants; water at the base."]),
    "Potato___Early_blight": ("fungal", "medium", "Fungus (Alternaria solani). Dark brown spots with concentric rings ('target' pattern) on older leaves first.",
        ["Rotate crops (2+ years away from potato/tomato).", "Keep plants well fed and watered - stressed plants are hit hardest.", "Apply a labelled protectant fungicide at first symptoms.", "Remove volunteer plants and crop debris."]),
    "Potato___Late_blight": ("fungal", "high", "Oomycete (Phytophthora infestans). Water-soaked dark lesions, white fuzz under leaves in humid weather; can destroy a field within days.",
        ["Act immediately - spread is very fast in cool, wet weather.", "Apply a protectant plus systemic fungicide as per local late-blight advice.", "Destroy infected plants and cull piles; do not compost them.", "Delay harvest until vines are dead to protect tubers, and use certified seed next season."]),
    "Squash___Powdery_mildew": ("fungal", "medium", "Fungus (Podosphaera xanthii). White, talc-like patches on leaf surfaces that spread and yellow the leaves.",
        ["Grow resistant varieties and give plants space.", "Spray sulfur, potassium bicarbonate or a labelled fungicide at first sign; repeat as directed.", "Water early in the day at the soil line.", "Remove badly affected leaves."]),
    "Strawberry___Leaf_scorch": ("fungal", "medium", "Fungus (Diplocarpon earlianum). Many small purple spots that enlarge and darken; leaves look scorched.",
        ["Remove old infected leaves after harvest (renovation).", "Use drip irrigation and wider spacing for airflow.", "Plant resistant cultivars and clean certified plants.", "Apply a labelled fungicide if disease is heavy."]),
    "Tomato___Bacterial_spot": ("bacterial", "medium", "Bacteria (Xanthomonas spp.). Small dark, greasy-looking spots on leaves and rough scabs on fruit.",
        ["Use clean seed and transplants.", "Copper-based sprays as labelled can slow spread.", "Rotate crops and remove debris.", "Avoid overhead watering and handling wet plants."]),
    "Tomato___Early_blight": ("fungal", "medium", "Fungus (Alternaria). Brown spots with concentric rings on lower leaves, often with yellow halos.",
        ["Remove lower infected leaves and stake plants to keep foliage off the soil.", "Mulch to stop soil splash.", "Apply a labelled fungicide at first symptoms.", "Rotate away from tomato/potato for 2 years."]),
    "Tomato___Late_blight": ("fungal", "high", "Oomycete (Phytophthora infestans). Large greasy gray-green blotches, white mould underneath in humid weather; fruit rots.",
        ["Act immediately - it spreads in days and also infects potatoes.", "Remove and bag/destroy infected plants (do not compost).", "Apply a labelled fungicide to healthy plants nearby.", "Avoid overhead irrigation; space and stake for airflow."]),
    "Tomato___Leaf_Mold": ("fungal", "medium", "Fungus (Passalora fulva). Pale yellow patches on the upper leaf surface with olive-gray mould underneath; common in humid greenhouses.",
        ["Lower humidity and increase ventilation.", "Space and prune plants for airflow; water at the base.", "Remove infected leaves.", "Use resistant varieties and a labelled fungicide if needed."]),
    "Tomato___Septoria_leaf_spot": ("fungal", "medium", "Fungus (Septoria lycopersici). Many small circular spots with gray centres and dark edges, starting on lower leaves.",
        ["Remove infected lower leaves early.", "Mulch and avoid splashing water on foliage.", "Apply a labelled protectant fungicide when spots first appear.", "Rotate crops for at least 2 years."]),
    "Tomato___Spider_mites Two-spotted_spider_mite": ("pest", "medium", "Two-spotted spider mites (not a disease). Fine yellow stippling, bronzing and webbing, worst in hot, dry, dusty weather.",
        ["Spray plants (especially leaf undersides) with a strong jet of water.", "Use insecticidal soap, neem or a labelled miticide; repeat as directed.", "Encourage or release predatory mites; avoid broad-spectrum insecticides that kill them.", "Keep plants watered and reduce dust."]),
    "Tomato___Target_Spot": ("fungal", "medium", "Fungus (Corynespora cassiicola). Brown lesions with concentric rings and yellow halos on leaves and fruit.",
        ["Improve airflow by staking and pruning.", "Remove infected leaves and crop debris.", "Apply a labelled fungicide when lesions appear.", "Rotate crops and avoid overhead irrigation."]),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("viral", "high", "Tomato yellow leaf curl virus - spread by whiteflies. Upward-curling, yellowing, stunted leaves and poor fruit set.",
        ["There is no cure - remove and destroy infected plants.", "Control whiteflies: yellow sticky traps, reflective mulch, insect-proof nets over seedlings, labelled insecticides.", "Plant tolerant/resistant varieties.", "Do not leave infected plants or weeds near new plantings."]),
    "Tomato___Tomato_mosaic_virus": ("viral", "high", "Tomato mosaic virus. Mottled light/dark green leaves, distorted growth; spreads on hands, tools and seed.",
        ["There is no cure - remove and destroy infected plants.", "Wash hands and disinfect tools between plants; avoid handling tobacco before touching plants.", "Use certified virus-free seed and resistant varieties.", "Clean up plant debris and control weeds."]),
}

HEALTHY_TIPS = [
    "No sign of disease detected - keep scouting leaves weekly, including the undersides.",
    "Water at the base in the morning so leaves dry quickly.",
    "Keep tools clean, remove plant debris and maintain good airflow between plants.",
    "Re-scan any leaf that develops spots, curling, mould or unusual colour.",
]

DISCLAIMER = ("HarvestGuard is a decision-support tool, not a lab diagnosis. Confirm serious or unusual cases "
              "with a local agricultural extension officer, and follow product labels and local regulations "
              "for any treatment.")


def parse_label(label: str) -> tuple[str, str, bool]:
    crop, _, cond = label.partition("___")
    crop = re.sub(r"\(.*?\)", "", crop).replace("_", " ").replace(",", "").strip()
    crop = re.sub(r"\s+", " ", crop).strip()
    cond_clean = re.sub(r"\(.*?\)", lambda m: " " + m.group(0), cond)
    cond_clean = cond_clean.replace("_", " ").strip()
    cond_clean = re.sub(r"\s+", " ", cond_clean)
    healthy = cond.lower() == "healthy"
    return crop, ("Healthy" if healthy else cond_clean), healthy


def advice_for(label: str) -> dict:
    crop, cond, healthy = parse_label(label)
    if healthy:
        return {"crop": crop, "condition": cond, "healthy": True, "kind": "healthy", "urgency": "low",
                "summary": f"This {crop.lower()} leaf looks healthy.", "actions": HEALTHY_TIPS,
                "disclaimer": DISCLAIMER}
    kind, urgency, summary, actions = D.get(label, ("unknown", "medium",
                                                    "Condition recognised by the model.",
                                                    ["Consult a local agricultural extension officer."]))
    return {"crop": crop, "condition": cond, "healthy": False, "kind": kind, "urgency": urgency,
            "summary": summary, "actions": actions, "disclaimer": DISCLAIMER}
