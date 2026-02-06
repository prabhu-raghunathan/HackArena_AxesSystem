# from shapely import wkt
# from shapely.geometry import MultiLineString, box
# import matplotlib.pyplot as plt
# import uuid
# import os
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# async def run_geometry_qa(uploaded_file):
#     text = (await uploaded_file.read()).decode("utf-8")
#
#     lines = []
#     buffer = ""
#
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         buffer += line + " "
#         if line.endswith(")"):
#             lines.append(wkt.loads(buffer))
#             buffer = ""
#
#     streets = MultiLineString([l.coords for l in lines])
#     street_buffer = streets.buffer(2.0)
#
#     # Example labels (synthetic for demo)
#     labels = [
#         box(7200-8, 8660-4, 7200+8, 8660+4),
#         box(7155-8, 8645-4, 7155+8, 8645+4)
#     ]
#
#     errors = []
#     bad_labels = []
#
#     for i, label in enumerate(labels):
#         if label.intersects(street_buffer):
#             severity = label.intersection(street_buffer).area / label.area
#             errors.append({
#                 "type": "label_street_overlap",
#                 "severity": round(severity, 2),
#                 "description": "Label overlaps street geometry"
#             })
#             bad_labels.append(label)
#
#     # ---- Visualization ----
#     fig, ax = plt.subplots(figsize=(6,6))
#
#     for line in streets.geoms:
#         x, y = line.xy
#         ax.plot(x, y, color="black")
#
#     for label in labels:
#         x, y = label.exterior.xy
#         ax.plot(x, y, color="green")
#
#     for label in bad_labels:
#         x, y = label.exterior.xy
#         ax.fill(x, y, color="red", alpha=0.5)
#
#     ax.set_title("Geometry QA – Label/Street Overlap")
#     ax.set_aspect("equal")
#
#     out_name = f"geometry_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     plt.savefig(out_path)
#     plt.close()
#
#     return f"/outputs/{out_name}", errors
#
# from shapely import wkt
# from shapely.geometry import MultiLineString
# import matplotlib.pyplot as plt
# from sklearn.ensemble import IsolationForest
# import uuid
# import os
# import numpy as np
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#
# async def run_geometry_qa(uploaded_file):
#     # -----------------------------
#     # 1. Load WKT geometries
#     # -----------------------------
#     text = (await uploaded_file.read()).decode("utf-8")
#
#     lines = []
#     buffer = ""
#
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         buffer += line + " "
#         if line.endswith(")"):
#             geom = wkt.loads(buffer)
#             lines.append(geom)
#             buffer = ""
#
#     # -----------------------------
#     # 2. Extract geometric features
#     # -----------------------------
#     features = []
#     for geom in lines:
#         features.append([
#             geom.length,
#             len(geom.coords)
#         ])
#
#     X = np.array(features)
#
#     # -----------------------------
#     # 3. ML-based anomaly detection
#     # -----------------------------
#     model = IsolationForest(
#         n_estimators=100,
#         contamination=0.15,
#         random_state=42
#     )
#     predictions = model.fit_predict(X)
#
#     # -----------------------------
#     # 4. Detect ONE error type
#     #    "Degenerate / Abnormally short lines"
#     # -----------------------------
#     errors = []
#     bad_indices = []
#
#     for idx, pred in enumerate(predictions):
#         if pred == -1:
#             severity = round(
#                 1.0 - (lines[idx].length / max(X[:, 0])),
#                 2
#             )
#
#             errors.append({
#                 "type": "degenerate_line_geometry",
#                 "geometry_index": idx,
#                 "severity": severity,
#                 "description": "Line geometry is abnormally short compared to peers"
#             })
#
#             bad_indices.append(idx)
#
#     # -----------------------------
#     # 5. Visualization (unchanged contract)
#     # -----------------------------
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     for i, geom in enumerate(lines):
#         x, y = geom.xy
#         if i in bad_indices:
#             ax.plot(x, y, color="red", linewidth=2)
#         else:
#             ax.plot(x, y, color="black", linewidth=1)
#
#     ax.set_title("Geometry QA – Degenerate Line Detection")
#     ax.set_aspect("equal")
#
#     out_name = f"geometry_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     plt.savefig(out_path)
#     plt.close()
#
#     return f"/outputs/{out_name}", errors


# from shapely import wkt
# from shapely.geometry import MultiLineString
# from shapely.validation import explain_validity
# import matplotlib.pyplot as plt
# import uuid
# import os
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#
# async def run_geometry_qa(uploaded_file):
#     # --------------------------------------------------
#     # 1. Load raw WKT geometries
#     # --------------------------------------------------
#     text = (await uploaded_file.read()).decode("utf-8")
#
#     lines = []
#     buffer = ""
#
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         buffer += line + " "
#         if line.endswith(")"):
#             geom = wkt.loads(buffer)
#             lines.append(geom)
#             buffer = ""
#
#     # --------------------------------------------------
#     # 2. RULE-BASED GEOMETRY VALIDATION (PRIMARY)
#     #    Single error type: invalid topology
#     # --------------------------------------------------
#     errors = []
#     invalid_indices = []
#
#     for idx, geom in enumerate(lines):
#         if not geom.is_valid:
#             errors.append({
#                 "type": "invalid_line_topology",
#                 "geometry_index": idx,
#                 "description": explain_validity(geom)
#             })
#             invalid_indices.append(idx)
#
#     # --------------------------------------------------
#     # 3. Visualization (unchanged integration contract)
#     # --------------------------------------------------
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     for i, geom in enumerate(lines):
#         x, y = geom.xy
#         if i in invalid_indices:
#             ax.plot(x, y, color="red", linewidth=2)
#         else:
#             ax.plot(x, y, color="black", linewidth=1)
#
#     ax.set_title("Geometry QA – Invalid Line Topology")
#     ax.set_aspect("equal")
#
#     out_name = f"geometry_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     plt.savefig(out_path)
#     plt.close()
#
#     return f"/outputs/{out_name}", errors

# from shapely import wkt
# from shapely.geometry import Point
# import matplotlib.pyplot as plt
# import uuid
# import os
# import math
#
# OUTPUT_DIR = "backend/outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# # Distance tolerance for snapping / connectivity
# TOLERANCE = 1.0
#
#
# async def run_geometry_qa(uploaded_file):
#     # --------------------------------------------------
#     # 1. Load raw WKT geometries
#     # --------------------------------------------------
#     text = (await uploaded_file.read()).decode("utf-8")
#
#     lines = []
#     buffer = ""
#
#     for line in text.splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         buffer += line + " "
#         if line.endswith(")"):
#             geom = wkt.loads(buffer)
#             lines.append(geom)
#             buffer = ""
#
#     # --------------------------------------------------
#     # 2. Collect all endpoints
#     # --------------------------------------------------
#     endpoints = []  # (Point, parent_line_index)
#
#     for idx, geom in enumerate(lines):
#         coords = list(geom.coords)
#         endpoints.append((Point(coords[0]), idx))        # start
#         endpoints.append((Point(coords[-1]), idx))       # end
#
#     # --------------------------------------------------
#     # 3. Dangling endpoint detection (SINGLE RULE)
#     # --------------------------------------------------
#     errors = []
#     dangling_points = []
#
#     for i, (pt, parent_idx) in enumerate(endpoints):
#         connected = False
#
#         for j, (other_pt, other_parent_idx) in enumerate(endpoints):
#             if i == j:
#                 continue
#
#             # Allow connection to any other line
#             if pt.distance(other_pt) <= TOLERANCE:
#                 connected = True
#                 break
#
#         if not connected:
#             dangling_points.append(pt)
#             errors.append({
#                 "type": "dangling_endpoint",
#                 "geometry_index": parent_idx,
#                 "description": "Line endpoint is not connected to any other line"
#             })
#
#     # --------------------------------------------------
#     # 4. Visualization (unchanged integration contract)
#     # --------------------------------------------------
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     # Draw all lines
#     for geom in lines:
#         x, y = geom.xy
#         ax.plot(x, y, color="black", linewidth=1)
#
#     # Highlight dangling endpoints
#     for pt in dangling_points:
#         ax.scatter(pt.x, pt.y, color="red", s=30)
#
#     ax.set_title("Geometry QA – Dangling Endpoint Detection")
#     ax.set_aspect("equal")
#
#     out_name = f"geometry_{uuid.uuid4().hex}.png"
#     out_path = os.path.join(OUTPUT_DIR, out_name)
#     plt.savefig(out_path)
#     plt.close()
#
#     return f"/outputs/{out_name}", errors
from shapely import wkt
from shapely.geometry import Point
from shapely.validation import explain_validity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import math
import uuid
import os

# --------------------------------------------------
# Config
# --------------------------------------------------
OUTPUT_DIR = "backend/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ENDPOINT_TOLERANCE = 1.0
ZSCORE_THRESHOLD = 3.0
IF_CONTAMINATION = 0.1


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def compute_angle(p_prev, p_curr, p_next):
    v1 = np.array([p_prev[0] - p_curr[0], p_prev[1] - p_curr[1]])
    v2 = np.array([p_next[0] - p_curr[0], p_next[1] - p_curr[1]])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0

    cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.acos(cos_t)


# --------------------------------------------------
# Feature extraction (single source of truth)
# --------------------------------------------------
def extract_features(lines):
    endpoints = []
    for i, g in enumerate(lines):
        coords = list(g.coords)
        endpoints.append((Point(coords[0]), i))
        endpoints.append((Point(coords[-1]), i))

    features = []

    for idx, g in enumerate(lines):
        coords = list(g.coords)
        n = len(coords)

        length = g.length
        avg_seg_len = length / (n - 1) if n > 1 else 0.0

        minx, miny, maxx, maxy = g.bounds
        bbox_area = (maxx - minx) * (maxy - miny)

        angles = []
        for i in range(1, n - 1):
            angles.append(compute_angle(coords[i - 1], coords[i], coords[i + 1]))
        curvature_var = float(np.var(angles)) if angles else 0.0

        endpoint_degree = 0
        for ep, parent in endpoints:
            if parent == idx:
                continue
            if (
                ep.distance(Point(coords[0])) <= ENDPOINT_TOLERANCE
                or ep.distance(Point(coords[-1])) <= ENDPOINT_TOLERANCE
            ):
                endpoint_degree += 1

        features.append({
            "geometry_index": idx,
            "length": length,
            "num_vertices": n,
            "avg_segment_length": avg_seg_len,
            "bbox_area": bbox_area,
            "curvature_variance": curvature_var,
            "endpoint_degree": endpoint_degree,
        })

    return features


# --------------------------------------------------
# Z-score anomaly detection
# --------------------------------------------------
# def zscore_anomalies(features, key):
#     vals = np.array([f[key] for f in features])
#     mean = vals.mean()
#     std = vals.std() if vals.std() != 0 else 1.0
#
#     out = []
#     for f in features:
#         z = abs((f[key] - mean) / std)
#         if z > ZSCORE_THRESHOLD:
#             out.append({
#                 "type": "geometric_anomaly_zscore",
#                 "geometry_index": f["geometry_index"],
#                 "feature": key,
#                 "zscore": round(z, 2),
#                 "description": f"Z-score anomaly on feature '{key}'"
#             })
#     return out
def zscore_anomalies(features, key):
    vals = np.array([f[key] for f in features])
    mean = vals.mean()
    std = vals.std() if vals.std() != 0 else 1.0

    out = []
    for f, z in zip(features, np.abs((vals - mean) / std)):
        if z > ZSCORE_THRESHOLD:
            out.append({
                "type": "geometric_anomaly_zscore",
                "geometry_index": f["geometry_index"],
                "severity": round(min(z / 6.0, 1.0), 2),
                "description": f"Z-score anomaly on feature '{key}'"
            })
    return out


# --------------------------------------------------
# Isolation Forest anomaly detection
# --------------------------------------------------
def isolation_forest_anomalies(features):
    X = np.array([
        [
            f["length"],
            f["num_vertices"],
            f["avg_segment_length"],
            f["bbox_area"],
            f["curvature_variance"],
            f["endpoint_degree"],
        ]
        for f in features
    ])

    model = IsolationForest(
        n_estimators=100,
        contamination=IF_CONTAMINATION,
        random_state=42
    )
    preds = model.fit_predict(X)

    out = []
    for f, p in zip(features, preds):
        if p == -1:
            out.append({
                "type": "geometric_anomaly_ml",
                "geometry_index": f["geometry_index"],
                "severity": 0.6,
                "description": "Isolation Forest detected unusual feature combination"
            })

            # out.append({
            #     "type": "geometric_anomaly_ml",
            #     "geometry_index": f["geometry_index"],
            #     "description": "Isolation Forest detected unusual feature combination"
            # })
    return out


# --------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------
async def run_geometry_qa(uploaded_file):
    # -------------------------------
    # 1. Load WKT
    # -------------------------------
    text = (await uploaded_file.read()).decode("utf-8")
    lines = []
    buf = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        buf += line + " "
        if line.endswith(")"):
            lines.append(wkt.loads(buf))
            buf = ""

        # -------------------------------
        # 2. INSUFFICIENT DATA GUARD
        # -------------------------------
    if len(lines) < 2:
            return None, [{
                "type": "insufficient_data",
                "severity": 1.0,
                "description": "At least two geometries are required for QA analysis"
            }]
    errors = []

    # -------------------------------
    # 2. RULE-BASED VALIDATION
    # -------------------------------
    for i, g in enumerate(lines):
        if not g.is_valid:
            errors.append({
                "type": "invalid_topology",
                "geometry_index": i,
                "severity": 1.0,
                "description": explain_validity(g)
            })

    # -------------------------------
    # 3. FEATURE EXTRACTION
    # -------------------------------
    features = extract_features(lines)

    # Dangling endpoints (rule-based)
    for f in features:
        if f["endpoint_degree"] == 0:
            errors.append({
                "type": "dangling_endpoint",
                "geometry_index": f["geometry_index"],
                "severity": 0.8,
                "description": "Line endpoint not connected to network"
            })

    # -------------------------------
    # 4. Z-SCORE ANOMALIES
    # -------------------------------
    for key in ["length", "num_vertices", "bbox_area"]:
        errors.extend(zscore_anomalies(features, key))

    # -------------------------------
    # 5. ISOLATION FOREST ANOMALIES
    # -------------------------------
    errors.extend(isolation_forest_anomalies(features))

    # -------------------------------
    # 6. Visualization
    # -------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    anomaly_ids = {e["geometry_index"] for e in errors}

    for i, g in enumerate(lines):
        x, y = g.xy
        if i in anomaly_ids:
            ax.plot(x, y, color="red", linewidth=2)
        else:
            ax.plot(x, y, color="black", linewidth=1)

    ax.set_title("Geometry QA – Rules + Z-score + Isolation Forest")
    ax.set_aspect("equal")

    out_name = f"geometry_{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path)
    plt.close()

    return f"/outputs/{out_name}", errors
