# Chương trình thu thập thủ công :), 
# Hạn chế sử dụng thư viện nếu có thể hoặc nếu dùng phải đảm bảo có ý nghĩa

import cv2
import os
import time
import mediapipe as mp
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

video = cv2.VideoCapture(1)
DATASET_DIR = "../dataset-razer"

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

MODEL_PATH = "hand_landmarker.task"

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


current_person_id = 1
left_dir, right_dir = "", ""
is_collecting = False
is_hand = 'Left'

img_counter = defaultdict(int)
img_counter['Left'] = 0
img_counter['Right'] = 0

print(f"\n=== CURRENT PERSON: person_{current_person_id} ===")
print("[SPACE] : Press for capturing")
print("[N]     : Next Person")
print("[L]     : Left")
print("[R]     : Right")
print("[Q]     : Quit")

while True:
    ret, display_frame = video.read()
    frame = display_frame.copy()

    if not ret:
        break

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

    cv2.imshow("Capture", display_frame)

    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):
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

    if key == ord('l'):
        is_hand = 'Left'
        print("Collect left hand mode!")

    if key == ord('r'):
        is_hand = 'Right'
        print("Collect right hand mode!")

    if key == 32:
        # Mã space
        is_collecting = True
    else:
        is_collecting = False

    if is_collecting:
        img_counter[is_hand] += 1
        
        save_dir = left_dir if is_hand == 'Left' else right_dir
        save_path = os.path.join(
            save_dir,
            f"{img_counter[is_hand]}.jpg"
        )

        cv2.imwrite(
            save_path,
            frame
        )

    if key == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
