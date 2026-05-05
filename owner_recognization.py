import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision import transforms
import torchreid

# ---------------------------------
# LOAD YOLO
# ---------------------------------
model = YOLO("yolov8n.pt")

# ---------------------------------
# DEEPSORT
# ---------------------------------
tracker = DeepSort(
    max_age=50,
    n_init=2,
    max_cosine_distance=0.3
)

# ---------------------------------
# LOAD REID MODEL
# ---------------------------------
reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
)

reid_model.eval()

# ---------------------------------
# IMAGE TRANSFORM
# ---------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
])

# ---------------------------------
# CAMERA
# ---------------------------------
cap = cv2.VideoCapture(0)

# ---------------------------------
# VARIABLES
# ---------------------------------
target_embedding = None
target_id = None
clicked_point = None

# ---------------------------------
# MOUSE CLICK
# ---------------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("AI PERSON LOCK")
cv2.setMouseCallback("AI PERSON LOCK", mouse_callback)

# ---------------------------------
# FEATURE EXTRACTOR
# ---------------------------------
def extract_features(person_img):

    img = transform(person_img).unsqueeze(0)

    with torch.no_grad():
        features = reid_model(img)

    return features.numpy()

# ---------------------------------
# COSINE SIMILARITY
# ---------------------------------
def cosine_similarity(a, b):

    a = a.flatten()
    b = b.flatten()

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# ---------------------------------
# LOOP
# ---------------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (900, 700))

    results = model(frame)

    detections = []

    # -----------------------------
    # DETECTIONS
    # -----------------------------
    for r in results:

        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])

                detections.append(
                    ([x1, y1, x2-x1, y2-y1], conf, 'person')
                )

    # -----------------------------
    # TRACKING
    # -----------------------------
    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    best_similarity = 0
    best_track = None

    # -----------------------------
    # PROCESS TRACKS
    # -----------------------------
    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l, t, r, b = map(int, track.to_ltrb())

        w = r - l
        h = b - t

        # Safety
        if w <= 0 or h <= 0:
            continue

        person_crop = frame[t:b, l:r]

        # Extract features
        features = extract_features(person_crop)

        # -----------------------------
        # USER CLICK LOCK
        # -----------------------------
        if clicked_point is not None:

            px, py = clicked_point

            if l < px < r and t < py < b:

                target_embedding = features
                target_id = track_id
                clicked_point = None

        color = (255, 0, 0)

        # -----------------------------
        # REID MATCHING
        # -----------------------------
        if target_embedding is not None:

            similarity = cosine_similarity(
                target_embedding,
                features
            )

            # Best visual match
            if similarity > best_similarity:
                best_similarity = similarity
                best_track = (
                    track_id,
                    l, t, r, b,
                    similarity
                )

        # Draw all
        cv2.rectangle(frame, (l, t), (r, b), color, 2)

        cv2.putText(
            frame,
            f"ID {track_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # --------------------------------
    # LOCKED TARGET
    # --------------------------------
    if best_track is not None:

        track_id, l, t, r, b, similarity = best_track

        # Similarity threshold
        if similarity > 0.75:

            cv2.rectangle(
                frame,
                (l, t),
                (r, b),
                (0, 255, 0),
                4
            )

            cv2.putText(
                frame,
                f"LOCKED | SIM: {similarity:.2f}",
                (l, t - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                3
            )

    # --------------------------------
    # TEXT
    # --------------------------------
    cv2.putText(
        frame,
        "Click Person To LOCK Identity",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI PERSON LOCK", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()