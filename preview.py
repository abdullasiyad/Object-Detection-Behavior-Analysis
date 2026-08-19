from ultralytics import YOLO
import cv2

# ============================================================
# SETTINGS - CHANGE ONLY THESE
# ============================================================

MODEL_PATH = "yolov8/weights/best.pt"
VIDEO_PATH = "videos/20230711a-fan.mp4"

CONFIDENCE = 0.50
SKIP_FRAMES = 1

# Colors use BGR format
BOX_COLOR = (0, 255, 0)        # GREEN
TEXT_COLOR = (255, 255, 255)   # White
TEXT_BACKGROUND = (0, 0, 0)    # Black

BOX_THICKNESS = 1
FONT_SIZE = 0.45
TEXT_THICKNESS = 1


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")
print("Classes:", model.names)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video:")
    print(VIDEO_PATH)
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

delay = max(1, int(1000 / fps))

print(f"Video FPS: {fps:.1f}")
print("Press Q to quit.")


# ============================================================
# PREVIEW
# ============================================================

frame_number = 0
detections = []

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    if frame_number % SKIP_FRAMES == 0:

        results = model(
            frame,
            imgsz=640,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0] if isinstance(results, list) else next(iter(results))

        detections = []

        boxes = getattr(result, "boxes", [])
        if boxes is not None:
            for box in boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                confidence = float(box.conf[0])

                detections.append(
                    (x1, y1, x2, y2, confidence)
                )


    # --------------------------------------------------------
    # DRAW DETECTIONS
    # --------------------------------------------------------

    for x1, y1, x2, y2, confidence in detections:

        label = f"bee {confidence:.2f}"

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            BOX_THICKNESS
        )

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            TEXT_THICKNESS
        )

        # Text background position
        text_top = max(0, y1 - text_height - 8)
        text_bottom = y1

        # Text background
        cv2.rectangle(
            frame,
            (x1, text_top),
            (x1 + text_width + 6, text_bottom),
            TEXT_BACKGROUND,
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SIZE,
            TEXT_COLOR,
            TEXT_THICKNESS,
            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # SHOW VIDEO
    # --------------------------------------------------------

    cv2.imshow(
        "Bee Detection Preview",
        frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(delay) & 0xFF == ord("q"):
        print("Preview stopped.")
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Done.")