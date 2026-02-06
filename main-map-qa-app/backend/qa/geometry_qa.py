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

from shapely import wkt
from shapely.geometry import Point
import matplotlib.pyplot as plt
import uuid
import os
import math

OUTPUT_DIR = "backend/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Distance tolerance for snapping / connectivity
TOLERANCE = 1.0


async def run_geometry_qa(uploaded_file):
    # --------------------------------------------------
    # 1. Load raw WKT geometries
    # --------------------------------------------------
    text = (await uploaded_file.read()).decode("utf-8")

    lines = []
    buffer = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        buffer += line + " "
        if line.endswith(")"):
            geom = wkt.loads(buffer)
            lines.append(geom)
            buffer = ""

    # --------------------------------------------------
    # 2. Collect all endpoints
    # --------------------------------------------------
    endpoints = []  # (Point, parent_line_index)

    for idx, geom in enumerate(lines):
        coords = list(geom.coords)
        endpoints.append((Point(coords[0]), idx))        # start
        endpoints.append((Point(coords[-1]), idx))       # end

    # --------------------------------------------------
    # 3. Dangling endpoint detection (SINGLE RULE)
    # --------------------------------------------------
    errors = []
    dangling_points = []

    for i, (pt, parent_idx) in enumerate(endpoints):
        connected = False

        for j, (other_pt, other_parent_idx) in enumerate(endpoints):
            if i == j:
                continue

            # Allow connection to any other line
            if pt.distance(other_pt) <= TOLERANCE:
                connected = True
                break

        if not connected:
            dangling_points.append(pt)
            errors.append({
                "type": "dangling_endpoint",
                "geometry_index": parent_idx,
                "description": "Line endpoint is not connected to any other line"
            })

    # --------------------------------------------------
    # 4. Visualization (unchanged integration contract)
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw all lines
    for geom in lines:
        x, y = geom.xy
        ax.plot(x, y, color="black", linewidth=1)

    # Highlight dangling endpoints
    for pt in dangling_points:
        ax.scatter(pt.x, pt.y, color="red", s=30)

    ax.set_title("Geometry QA – Dangling Endpoint Detection")
    ax.set_aspect("equal")

    out_name = f"geometry_{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    plt.savefig(out_path)
    plt.close()

    return f"/outputs/{out_name}", errors
