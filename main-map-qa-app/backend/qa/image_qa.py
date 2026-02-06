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
import cv2
import numpy as np
import uuid
import os

OUTPUT_DIR = "backend/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
Training instructions:
1. Collect 5–10 screenshots of CORRECT map outputs
2. For each image, compute:
   - edge_density
   - line_count
3. Set the minimum acceptable values below based on those examples
"""

# Learned baseline from "good" images (example values)
GOOD_IMAGE_BASELINE = {
    "min_edge_density": 0.015,
    "min_line_count": 15
}


async def run_image_qa(uploaded_file):
    # --------------------------------------------------
    # 1. Load image
    # --------------------------------------------------
    file_bytes = np.frombuffer(await uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, [{
            "type": "invalid_image",
            "confidence": 1.0,
            "description": "Uploaded file is not a valid image"
        }]

    original = img.copy()

    # --------------------------------------------------
    # 2. Preprocessing
    # --------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --------------------------------------------------
    # 3. Edge detection
    # --------------------------------------------------
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

    h, w = edges.shape
    edge_density = np.sum(edges > 0) / (h * w)

    # --------------------------------------------------
    # 4. Line detection (Hough Transform)
    # --------------------------------------------------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=40,
        maxLineGap=15
    )

    line_count = 0
    if lines is not None:
        line_count = len(lines)

    # --------------------------------------------------
    # 5. Rule + learned baseline comparison (AI-assisted)
    # --------------------------------------------------
    errors = []

    if (
        edge_density < GOOD_IMAGE_BASELINE["min_edge_density"]
        or line_count < GOOD_IMAGE_BASELINE["min_line_count"]
    ):
        confidence = round(
            1.0 - min(
                edge_density / GOOD_IMAGE_BASELINE["min_edge_density"],
                line_count / GOOD_IMAGE_BASELINE["min_line_count"],
                1.0
            ),
            2
        )

        errors.append({
            "type": "missing_or_broken_roads",
            "confidence": confidence,
            "description": "Detected low road structure compared to trained baseline"
        })

    # --------------------------------------------------
    # 6. Visualization (show WHERE the issue is)
    # --------------------------------------------------
    vis = original.copy()

    if lines is not None:
        xs, ys = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            xs.extend([x1, x2])
            ys.extend([y1, y2])

        if errors:
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            cv2.rectangle(
                vis,
                (min_x, min_y),
                (max_x, max_y),
                (0, 0, 255),
                2
            )

    if errors:
        cv2.putText(
            vis,
            "Missing / Broken Roads Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # --------------------------------------------------
    # 7. Save output
    # --------------------------------------------------
    out_name = f"image_{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, vis)

    return f"/outputs/{out_name}", errors
