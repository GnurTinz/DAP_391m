import os
import sys
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# =========================================================
# 1. CONFIG
# =========================================================

def resource_path(relative_path):

    try:
        base_path = sys._MEIPASS

    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(
        base_path,
        relative_path
    )

# MODEL_PATH = resource_path(
#     "hand_landmarker.task"
# )

MODEL_PATH = "hand_landmarker.task"
DATASET_DIR = "dataset"

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7
PRESENCE_CONFIDENCE = 0.7

PADDING = 30

# Làm mờ nền
BLUR_BACKGROUND = True
BLUR_STRENGTH = 35

os.makedirs(DATASET_DIR, exist_ok=True)

# =========================================================
# 2. MEDIAPIPE TASKS API
# =========================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=MAX_HANDS,
    min_hand_detection_confidence=DETECTION_CONFIDENCE,
    min_tracking_confidence=TRACKING_CONFIDENCE,
    min_hand_presence_confidence=PRESENCE_CONFIDENCE,
    running_mode=vision.RunningMode.VIDEO,
)

detector = vision.HandLandmarker.create_from_options(options)

# =========================================================
# 3. AUTO PERSON ID
# =========================================================

def get_next_person_id(base_dir):

    ids = []

    for folder in os.listdir(base_dir):

        if folder.startswith("person_"):

            try:
                ids.append(
                    int(folder.split("_")[1])
                )

            except:
                pass

    return max(ids) + 1 if ids else 1


def setup_person_folder(person_id):

    person_folder = os.path.join(
        DATASET_DIR,
        f"person_{person_id}"
    )

    left_dir = os.path.join(
        person_folder,
        "left"
    )

    right_dir = os.path.join(
        person_folder,
        "right"
    )

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    counter = {
        "Left": len(os.listdir(left_dir)),
        "Right": len(os.listdir(right_dir)),
    }

    return left_dir, right_dir, counter


current_person_id = get_next_person_id(
    DATASET_DIR
)

left_dir, right_dir, img_counter = setup_person_folder(
    current_person_id
)

# =========================================================
# 4. CAMERA
# =========================================================

cap = cv2.VideoCapture(1
)

is_collecting = False
timestamp = 0

print(f"\n=== CURRENT PERSON: person_{current_person_id} ===")
print("[SPACE] : Start / Pause")
print("[N]     : Next Person")
print("[Q]     : Quit")

# =========================================================
# 5. MAIN LOOP
# =========================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    original_frame = frame.copy()

    h, w, _ = frame.shape

    # =====================================================
    # RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # =====================================================
    # DETECTION
    # =====================================================

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 1

    # =====================================================
    # BACKGROUND BLUR
    # =====================================================

    if BLUR_BACKGROUND:

        blurred = cv2.GaussianBlur(
            frame,
            (BLUR_STRENGTH, BLUR_STRENGTH),
            0
        )

        display_frame = blurred.copy()

    else:
        display_frame = frame.copy()

    # =====================================================
    # HAND PROCESSING
    # =====================================================

    if result.hand_landmarks:

        for idx, hand_landmarks in enumerate(
            result.hand_landmarks
        ):

            handedness = result.handedness[idx][0].category_name

            # =================================================
            # LANDMARK -> PIXELS
            # =================================================

            x_points = [
                int(lm.x * w)
                for lm in hand_landmarks
            ]

            y_points = [
                int(lm.y * h)
                for lm in hand_landmarks
            ]

            # =================================================
            # BOUNDING BOX
            # =================================================

            x_min = max(min(x_points) - PADDING, 0)
            y_min = max(min(y_points) - PADDING, 0)

            x_max = min(max(x_points) + PADDING, w)
            y_max = min(max(y_points) + PADDING, h)

            # # =================================================
            # # KHÔI PHỤC VÙNG TAY KHÔNG BỊ MỜ
            # # =================================================

            # display_frame[
            #     y_min:y_max,
            #     x_min:x_max
            # ] = original_frame[
            #     y_min:y_max,
            #     x_min:x_max
            # ]

            # =================================================
            # TẠO MASK BÀN TAY
            # =================================================

            hand_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            # Landmark -> polygon points
            points = np.array(
                [
                    [int(lm.x * w), int(lm.y * h)]
                    for lm in hand_landmarks
                ],
                dtype=np.int32
            )

            # Convex hull để tạo vùng bàn tay kín
            hull = cv2.convexHull(points)

            # Fill vùng bàn tay
            cv2.fillConvexPoly(
                hand_mask,
                hull,
                255
            )

            # Làm mềm biên mask
            hand_mask = cv2.GaussianBlur(
                hand_mask,
                (21, 21),
                0
            )

            # =================================================
            # CHỈ GIỮ TAY KHÔNG BỊ BLUR
            # =================================================

            mask_3ch = cv2.merge(
                [hand_mask, hand_mask, hand_mask]
            ) / 255.0

            display_frame = (
                original_frame * mask_3ch +
                display_frame * (1 - mask_3ch)
            ).astype(np.uint8)

            # =================================================
            # DRAW LANDMARKS
            # =================================================

            for lm in hand_landmarks:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(
                    display_frame,
                    (cx, cy),
                    3,
                    (0, 255, 0),
                    -1
                )

            # =================================================
            # DRAW BBOX
            # =================================================

            cv2.rectangle(
                display_frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                handedness,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # =================================================
            # SAVE IMAGE
            # =================================================

            if is_collecting:

                palm_crop = original_frame[
                    y_min:y_max,
                    x_min:x_max
                ]

                if palm_crop.size > 0:

                    img_counter[handedness] += 1

                    save_dir = (
                        left_dir
                        if handedness == "Left"
                        else right_dir
                    )

                    save_path = os.path.join(
                        save_dir,
                        f"{img_counter[handedness]}.jpg"
                    )

                    cv2.imwrite(
                        save_path,
                        palm_crop
                    )
                
                if img_counter[handedness] >= 30:
                    is_collecting = False

    # =====================================================
    # UI
    # =====================================================

    status = (
        "COLLECTING"
        if is_collecting
        else "PAUSED"
    )

    status_color = (
        (0, 0, 255)
        if is_collecting
        else (255, 0, 0)
    )

    cv2.putText(
        display_frame,
        f"ID: person_{current_person_id}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display_frame,
        f"STATUS: {status}",
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        display_frame,
        f"Left: {img_counter['Left']} | Right: {img_counter['Right']}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Hand Collector + Blur Background",
        display_frame
    )

    # =====================================================
    # KEYBOARD
    # =====================================================

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        is_collecting = not is_collecting

        print(
            f"Collecting: {is_collecting}"
        )

    # NEXT PERSON
    elif key == ord("n"):

        is_collecting = False

        current_person_id = get_next_person_id(
            DATASET_DIR
        )

        left_dir, right_dir, img_counter = setup_person_folder(
            current_person_id
        )

        print(
            f"\n=== SWITCH TO person_{current_person_id} ==="
        )

    # QUIT
    elif key == ord("q"):

        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()

cv2.destroyAllWindows()

detector.close()