import os
import cv2
import time
import shutil
from ultralytics import YOLO
import gradio as gr

print("RUNNING FILE:", os.path.abspath(__file__))
print("🔥 RUNNING UPDATED GRADIO APP")

# ---------------- LOAD MODEL ----------------
model = YOLO("../runs/detect/train-5/weights/best.pt")

# ---------------- PATHS ----------------
INPUT_IMG_DIR = "inputs/image"
INPUT_VID_DIR = "inputs/video"
OUTPUT_IMG_DIR = "outputs/image"
OUTPUT_VID_DIR = "outputs/video"

# Create folders
for path in [INPUT_IMG_DIR, INPUT_VID_DIR, OUTPUT_IMG_DIR, OUTPUT_VID_DIR]:
    os.makedirs(path, exist_ok=True)


# ---------------- DRAW BOXES ----------------
def draw_boxes(image, results):
    annotated = image.copy()

    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)

            label = f"WEAPON {conf:.2f}"

            # 🔴 Red color (BGR)
            color = (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4)

            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
            )

            # Draw filled rectangle for label background
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )

            # Put white text on red background
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                3
            )

    return annotated

# ---------------- IMAGE DETECTION ----------------
def detect_image(image):
    print("👉 detect_image called")
    print("MANUAL DRAW EXECUTED")

    timestamp = int(time.time())

    input_path = os.path.join(INPUT_IMG_DIR, f"input_{timestamp}.jpg")
    output_path = os.path.join(OUTPUT_IMG_DIR, f"output_{timestamp}.jpg")

    # Save input
    cv2.imwrite(input_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Run model
    results = model(image)

    # Draw boxes manually
    annotated = draw_boxes(image, results)

    # Save output
    cv2.imwrite(output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # ✅ IMPORTANT FIX
    return annotated, output_path


# ---------------- VIDEO DETECTION ----------------
def detect_video(video_file):
    print("👉 detect_video called")

    timestamp = int(time.time())

    input_path = os.path.join(INPUT_VID_DIR, f"input_{timestamp}.mp4")
    output_path = os.path.join(OUTPUT_VID_DIR, f"output_{timestamp}.mp4")

    # ✅ safer than rename
    shutil.copy(video_file, input_path)

    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)
        annotated = draw_boxes(frame, results)

        if out is None:
            h, w, _ = annotated.shape
            out = cv2.VideoWriter(output_path, fourcc, 20, (w, h))

        out.write(annotated)

    cap.release()
    if out:
        out.release()

    # ✅ IMPORTANT FIX
    return output_path


# ---------------- GRADIO UI ----------------
with gr.Blocks() as demo:
    gr.Markdown("# 🛡️ Weapon Detection System (YOLOv26m)")

    # -------- IMAGE --------
    with gr.Tab("Image Detection"):
        img_input = gr.Image(type="numpy", label="Upload Image")
        img_output = gr.Image(label="Detected Image")
        img_download = gr.File(label="Download Output")

        btn_img = gr.Button("Run Detection")
        btn_img.click(
            detect_image,
            inputs=img_input,
            outputs=[img_output, img_download]
        )

    # -------- VIDEO --------
    with gr.Tab("Video Detection"):
        vid_input = gr.Video(label="Upload Video")
        vid_output = gr.Video(label="Detected Video")

        btn_vid = gr.Button("Run Detection")
        btn_vid.click(
            detect_video,
            inputs=vid_input,
            outputs=vid_output
        )


# ---------------- RUN ----------------
if __name__ == "__main__":
    demo.launch(debug=True)