import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import cv2
import torch
import numpy as np
import os
from collections import defaultdict
from torchvision import transforms, models
from torch import nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from ultralytics import YOLO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO(
    os.path.join(BASE_DIR, "yolov8n.pt")
) 


num_classes = 5

model = convnext_tiny(weights=None)
in_features = model.classifier[2].in_features

model.classifier[2] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, num_classes)
)

checkpoint = torch.load(
    os.path.join(BASE_DIR, "snake_model_convnext_tiny.pth"),
    map_location=device
)
model.load_state_dict(checkpoint["model_state_dict"])

model = model.to(device)
model.eval()


weights = ConvNeXt_Tiny_Weights.DEFAULT
transform = weights.transforms()

class_map = {
    0: "Nerodia sipedon - Northern Watersnake",
    1: "Thamnophis sirtalis - Common Garter snake",
    2: "Storeria dekayi - DeKay's Brown snake",
    3: "Pantherophis obsoletus - Black Rat snake",
    4: "Crotalus atrox - Western Diamondback rattlesnake"
}



def classify_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    pred_class = pred.item()
    return class_map[pred_class], conf.item(), pred_class

def extract_frames(video_path, fps_skip=5):
    cap = cv2.VideoCapture(video_path)
    frames = []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    step = max(1, fps // fps_skip)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % step == 0:
            frames.append(frame)

        count += 1

    cap.release()
    return frames


@app.get("/")
def root():
    return {
        "status": "online",
        "message": "Snake Detection API is running"
    }
    
    
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    
    
    image = Image.open(file.file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )
    
    label, conf, class_id = classify_image(image_np)

    return {
        "prediction": label,
        "class_id": class_id,
        "confidence": float(conf)
    }

@app.post("/predict-video")
async def predict_video(
    file: UploadFile = File(...)
):

    
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as tmp:

        shutil.copyfileobj(
            file.file,
            tmp
        )

        video_path = tmp.name

    try:

        frames = extract_frames(
            video_path
        )

        predictions = []

        for frame in frames:

            results = yolo_model(frame)

            detected = False

            for r in results:

                boxes = (
                    r.boxes.xyxy.cpu().numpy()
                    if r.boxes is not None
                    else []
                )

                if len(boxes) > 0:

                    detected = True

                    
                    areas = [
                        (b[2] - b[0]) *
                        (b[3] - b[1])
                        for b in boxes
                    ]

                    box = boxes[
                        np.argmax(areas)
                    ]

                    x1, y1, x2, y2 = map(
                        int,
                        box[:4]
                    )

                    crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    if (
                        crop is None
                        or crop.size == 0
                    ):
                        continue

                    label, conf, class_id = classify_image(
                        crop
                    )

                    predictions.append(
                        (
                            label,
                            conf,
                            class_id
                        )
                    )

            
            if not detected:

                label, conf, class_id = classify_image(
                    frame
                )

                predictions.append(
                    (
                        label,
                        conf,
                        class_id
                    )
                )

        if predictions:

            vote_dict = defaultdict(float)

            for label, conf, _ in predictions:

                vote_dict[label] += conf

            final_label = max(
                vote_dict,
                key=vote_dict.get
            )

            final_class_id = next(
                p[2]
                for p in predictions
                if p[0] == final_label
            )

            avg_conf = (
                vote_dict[final_label]
                / len(predictions)
            )

            return {
                "prediction": final_label,
                "class_id": final_class_id,
                "confidence": float(avg_conf),
                "frames_processed": len(frames),
                "detections": len(predictions)
            }

        return {
            "prediction": "No snake detected",
            "confidence": 0.0,
            "frames_processed": len(frames),
            "detections": 0
        }

    finally:

        if os.path.exists(video_path):

            os.remove(video_path)