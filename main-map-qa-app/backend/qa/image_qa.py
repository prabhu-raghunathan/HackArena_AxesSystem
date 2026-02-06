# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import uuid
# import os
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# async def run_image_qa(uploaded_file):
#     # ---- Load image ----
#     file_bytes = np.frombuffer(await uploaded_file.read(), np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#     if img is None:
#         return None, [{
#             "type": "invalid_image",
#             "confidence": 1.0,
#             "description": "Uploaded file is not a valid image"
#         }]
#
#     original = img.copy()
#
#     # ---- Step 1: Preprocessing ----
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # ---- Step 2: Edge Detection ----
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#     # ---- Step 3: Line Detection (Hough Transform) ----
#     lines = cv2.HoughLinesP(
#         edges,
#         rho=1,
#         theta=np.pi / 180,
#         threshold=100,
#         minLineLength=40,
#         maxLineGap=15
#     )
#
#     # ---- Step 4: QA Heuristics ----
#     h, w = edges.shape
#     edge_density = np.sum(edges > 0) / (h * w)
#
#     line_count = 0
#     if lines is not None:
#         line_count = len(lines)
#
#     # Simple anomaly rule
#     anomaly_score = 1.0 - min(edge_density * 5, 1.0)
#
#     errors = []
#
#     if edge_density < 0.01 or line_count < 10:
#         errors.append({
#             "type": "missing_or_broken_roads",
#             "confidence": round(anomaly_score, 2),
#             "description": "Low road structure detected (possible missing or broken roads)"
#         })
#
#     # ---- Step 5: Visualization ----
#     vis = original.copy()
#
#     if lines is not None:
#         for l in lines:
#             x1, y1, x2, y2 = l[0]
#             cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     if errors:
#         cv2.putText(
#             vis,
#             "Possible road anomaly detected",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )
#
#     # ---- Save output ----
#     out_name = f"image_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     cv2.imwrite(out_path, vis)
#
#     return f"/outputs/{out_name}", errors
#
#
# import cv2
# import numpy as np
# import uuid
# import os
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# """
# Training instructions:
# 1. Collect 5–10 screenshots of CORRECT map outputs
# 2. For each image, compute:
#    - edge_density
#    - line_count
# 3. Set the minimum acceptable values below based on those examples
# """
#
# # Learned baseline from "good" images (example values)
# GOOD_IMAGE_BASELINE = {
#     "min_edge_density": 0.015,
#     "min_line_count": 15
# }
#
#
# async def run_image_qa(uploaded_file):
#     # --------------------------------------------------
#     # 1. Load image
#     # --------------------------------------------------
#     file_bytes = np.frombuffer(await uploaded_file.read(), np.uint8)
#     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#
#     if img is None:
#         return None, [{
#             "type": "invalid_image",
#             "confidence": 1.0,
#             "description": "Uploaded file is not a valid image"
#         }]
#
#     original = img.copy()
#
#     # --------------------------------------------------
#     # 2. Preprocessing
#     # --------------------------------------------------
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     # --------------------------------------------------
#     # 3. Edge detection
#     # --------------------------------------------------
#     edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
#
#     h, w = edges.shape
#     edge_density = np.sum(edges > 0) / (h * w)
#
#     # --------------------------------------------------
#     # 4. Line detection (Hough Transform)
#     # --------------------------------------------------
#     lines = cv2.HoughLinesP(
#         edges,
#         rho=1,
#         theta=np.pi / 180,
#         threshold=100,
#         minLineLength=40,
#         maxLineGap=15
#     )
#
#     line_count = 0
#     if lines is not None:
#         line_count = len(lines)
#
#     # --------------------------------------------------
#     # 5. Rule + learned baseline comparison (AI-assisted)
#     # --------------------------------------------------
#     errors = []
#
#     if (
#         edge_density < GOOD_IMAGE_BASELINE["min_edge_density"]
#         or line_count < GOOD_IMAGE_BASELINE["min_line_count"]
#     ):
#         confidence = round(
#             1.0 - min(
#                 edge_density / GOOD_IMAGE_BASELINE["min_edge_density"],
#                 line_count / GOOD_IMAGE_BASELINE["min_line_count"],
#                 1.0
#             ),
#             2
#         )
#
#         errors.append({
#             "type": "missing_or_broken_roads",
#             "confidence": confidence,
#             "description": "Detected low road structure compared to trained baseline"
#         })
#
#     # --------------------------------------------------
#     # 6. Visualization (show WHERE the issue is)
#     # --------------------------------------------------
#     vis = original.copy()
#
#     if lines is not None:
#         xs, ys = [], []
#         for l in lines:
#             x1, y1, x2, y2 = l[0]
#             cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             xs.extend([x1, x2])
#             ys.extend([y1, y2])
#
#         if errors:
#             min_x, max_x = min(xs), max(xs)
#             min_y, max_y = min(ys), max(ys)
#             cv2.rectangle(
#                 vis,
#                 (min_x, min_y),
#                 (max_x, max_y),
#                 (0, 0, 255),
#                 2
#             )
#
#     if errors:
#         cv2.putText(
#             vis,
#             "Missing / Broken Roads Detected",
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             1,
#             (0, 0, 255),
#             2
#         )
#
#     # --------------------------------------------------
#     # 7. Save output
#     # --------------------------------------------------
#     out_name = f"image_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     cv2.imwrite(out_path, vis)
#
#     return f"/outputs/{out_name}", errors

import cv2
import numpy as np
import pickle
import uuid
import os
from sklearn.ensemble import IsolationForest

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
OUTPUT_DIR = "backend/outputs"
BASELINE_PATH = "backend/image_baseline.pkl"
GOOD_IMAGE_DIR = "good_images"  # folder with baseline screenshots

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------
def extract_image_features(img):
    # ---- Preprocess ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---- Edge Detection ----
    edges = cv2.Canny(blurred, 50, 150)
    h, w = edges.shape
    edge_density = np.sum(edges > 0) / (h * w)

    # ---- Line Detection ----
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=15
    )

    line_count = 0
    avg_line_len = 0.0
    orientations = []

    if lines is not None:
        line_count = len(lines)
        lengths = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            lengths.append(np.hypot(x2 - x1, y2 - y1))
            orientations.append(np.arctan2((y2 - y1), (x2 - x1)))
        avg_line_len = np.mean(lengths) if lengths else 0.0

    # ---- Connected Components (edges) ----
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    connected_components = num_labels - 1  # exclude background

    # ---- Coverage (rows & columns) ----
    row_coverage = np.mean(np.any(edges > 0, axis=1))
    col_coverage = np.mean(np.any(edges > 0, axis=0))

    # ---- Orientation Variance ----
    orientation_variance = float(np.var(orientations)) if orientations else 0.0

    return [
        edge_density,
        line_count,
        avg_line_len,
        orientation_variance,
        connected_components,
        row_coverage,
        col_coverage
    ]


# --------------------------------------------------
# Baseline Builder
# --------------------------------------------------
def build_baseline():
    """
    Read images from GOOD_IMAGE_DIR,
    compute features, then set:
      BASE_MEAN, BASE_STD, IF_MODEL
    and save to BASELINE_PATH
    """
    feature_list = []
    if not os.path.isdir(GOOD_IMAGE_DIR):
        return None, None, None

    for fname in os.listdir(GOOD_IMAGE_DIR):
        path = os.path.join(GOOD_IMAGE_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        feature_list.append(extract_image_features(img))

    if not feature_list:
        return None, None, None

    X = np.array(feature_list)
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # train simple IsolationForest
    if_model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    if_model.fit(X)

    # save baseline
    baseline = {"mean": mean, "std": std, "if_model": if_model}
    try:
        with open(BASELINE_PATH, "wb") as f:
            pickle.dump(baseline, f)
    except:
        pass

    return mean, std, if_model


# --------------------------------------------------
# Load or build baseline once
# --------------------------------------------------
BASE_MEAN, BASE_STD, IF_MODEL = None, None, None

if os.path.exists(BASELINE_PATH):
    try:
        with open(BASELINE_PATH, "rb") as f:
            baseline = pickle.load(f)
            BASE_MEAN = baseline.get("mean")
            BASE_STD = baseline.get("std")
            IF_MODEL = baseline.get("if_model")
    except Exception as e:
        # failed to load -> rebuild
        BASE_MEAN, BASE_STD, IF_MODEL = build_baseline()
else:
    BASE_MEAN, BASE_STD, IF_MODEL = build_baseline()


# --------------------------------------------------
# Anomaly Detection Helpers
# --------------------------------------------------
ZSCORE_THRESHOLD = 3.0

FEATURE_NAMES = [
    "edge_density",
    "line_count",
    "avg_line_length",
    "orientation_variance",
    "connected_components",
    "row_coverage",
    "col_coverage"
]

def zscore_anomalies_vector(features):
    if BASE_MEAN is None or BASE_STD is None:
        return []

    features = np.array(features)
    z = np.abs((features - BASE_MEAN) / (BASE_STD + 1e-9))

    anomalies = []
    for idx, val in enumerate(z):
        feature_name = FEATURE_NAMES[idx]

        # if baseline std is very small, use % deviation
        if BASE_STD[idx] < 0.01:
            diff = abs(features[idx] - BASE_MEAN[idx])
            perc = diff / (BASE_MEAN[idx] + 1e-9)

            if perc > 0.2:  # 20% baseline deviation
                anomalies.append({
                    "type": "image_zscore_anomaly",
                    "feature": feature_name,
                    "severity": round(min(perc, 1.0), 2),
                    "description": f"{feature_name} deviates {round(perc*100,1)}% from baseline"
                })
        else:
            # normal z-score based detection
            if val > ZSCORE_THRESHOLD:
                severity = round(min(val / (ZSCORE_THRESHOLD * 2), 1.0), 2)
                anomalies.append({
                    "type": "image_zscore_anomaly",
                    "feature": feature_name,
                    "severity": severity,
                    "description": f"{feature_name} Z-score {round(float(val),2)}"
                })
    return anomalies


def isolation_forest_anomaly(features):
    if IF_MODEL is None:
        return []

    X = np.array(features).reshape(1, -1)
    pred = IF_MODEL.predict(X)

    if pred == -1:
        return [{
            "type": "image_ml_anomaly",
            "severity": 0.6,
            "description": "Isolation Forest flagged this image as unusual"
        }]
    return []


# --------------------------------------------------
# MAIN ENTRY POINT for FastAPI
# --------------------------------------------------
async def run_image_qa(uploaded_file):
    # load image
    file_bytes = np.frombuffer(await uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, [{
            "type": "invalid_image",
            "severity": 1.0,
            "description": "Uploaded file is not a valid image"
        }]

    # compute features
    features = extract_image_features(img)

    # anomaly detection
    errors = []
    errors.extend(zscore_anomalies_vector(features))
    errors.extend(isolation_forest_anomaly(features))

    # visualize
    vis = img.copy()
    if errors:
        h, w, _ = vis.shape
        cv2.rectangle(vis, (0,0), (w-1,h-1), (0,0,255), 4)
        cv2.putText(
            vis,
            "Anomaly Detected",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    out_name = f"image_{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, vis)

    return f"/outputs/{out_name}", errors
