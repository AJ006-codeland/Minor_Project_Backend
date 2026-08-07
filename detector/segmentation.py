import os
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
WORKSPACE_NAME = "aayushmas-workspace-k2yif"
WORKFLOW_ID = "general-segmentation-api-4"
CLASSES = ["Leaf"]
MIN_AREA_RATIO = 0.005

INFERENCE_SERVER_URL = os.environ.get("INFERENCE_SERVER_URL", "http://localhost:9001")

client = InferenceHTTPClient(
    api_url=INFERENCE_SERVER_URL,
    api_key=API_KEY
)

def get_largest_leaf_crop(image_path, min_area_ratio=MIN_AREA_RATIO):
    """
    Runs the segmentation workflow on the given image path,
    and returns the single LARGEST leaf crop as a numpy array (BGR),
    or None if no valid leaf was found.
    """
    result = client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        parameters={"classes": CLASSES},
        use_cache=True
    )

    output = result[0] if isinstance(result, list) else result
    predictions = output.get("predictions")
    if not predictions:
        return None

    preds_list = predictions.get("predictions", predictions) if isinstance(predictions, dict) else predictions
    if not isinstance(preds_list, list) or len(preds_list) == 0:
        return None

    original_image = cv2.imread(image_path)
    img_area = original_image.shape[0] * original_image.shape[1]
    min_pixel_area = img_area * min_area_ratio

    best_crop = None
    best_area = 0

    for pred in preds_list:
        points = pred.get("points")
        if not points:
            continue

        poly = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)
        area = cv2.contourArea(poly)

        if area < min_pixel_area or area <= best_area:
            continue

        x, y, w, h = cv2.boundingRect(poly)
        mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        isolated = cv2.bitwise_and(original_image, original_image, mask=mask)
        crop = isolated[y:y + h, x:x + w]

        if crop.size == 0:
            continue

        best_crop = crop
        best_area = area

    return best_crop