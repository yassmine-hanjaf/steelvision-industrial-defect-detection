
import base64
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolo11n_neu_finetuned"
    / "weights"
    / "best.pt"
)

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Détection de défauts industriels avec YOLO11n entraîné sur NEU-DET.",
    version="1.1.0",
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Modèle YOLO introuvable : {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {
        "message": "Industrial Defect Detection API",
        "status": "running",
        "model": MODEL_PATH.name,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé est vide.",
        )

    try:
        image = Image.open(BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="Impossible de lire cette image.",
        )

    try:
        results = model.predict(
            source=image,
            imgsz=320,
            conf=0.25,
            verbose=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="La prédiction a échoué.",
        ) from exc

    # Créer une copie de l'image pour dessiner les détections
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()

    detections = []

    for result in results:
        boxes = result.boxes

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = (
                float(value)
                for value in box.xyxy[0].tolist()
            )

            class_name = model.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                },
            })

            # Dessiner le rectangle autour du défaut
            coordinates = (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            )

            draw.rectangle(
                coordinates,
                outline="#00E676",
                width=3,
            )

            # Ajouter le nom de la classe et la confiance
            label = f"{class_name} {confidence:.0%}"

            text_box = draw.textbbox(
                (int(x1), int(y1)),
                label,
                font=font,
            )

            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]

            label_y = max(0, int(y1) - text_height - 6)

            draw.rectangle(
                [
                    int(x1),
                    label_y,
                    int(x1) + text_width + 8,
                    label_y + text_height + 6,
                ],
                fill="#00A65A",
            )

            draw.text(
                (int(x1) + 4, label_y + 3),
                label,
                fill="white",
                font=font,
            )

    # Convertir l'image annotée en PNG puis en texte Base64
    image_buffer = BytesIO()
    annotated_image.save(image_buffer, format="PNG")

    annotated_image_base64 = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    return JSONResponse({
        "filename": file.filename,
        "image_width": image.width,
        "image_height": image.height,
        "number_of_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated_image_base64,
    })
