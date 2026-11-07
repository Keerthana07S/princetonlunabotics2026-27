import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from transformers import pipeline

# folders next to this script, so it works no matter where you run it from
HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
# The program will use this path to save the images with the
# identified objects and a csv file with the prediction values
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
# cutoff value to filter weak or uncertain predictions
THRESHOLD = 0.2

# this assumes all image file names begin with img_, e.g. img_001.[extension]
paths = sorted(IMAGES.glob("img_*"))
if not paths:
    raise SystemExit(f"No images starting with img_ found in {IMAGES}")

# set the zero-shot model to use
MODEL = "google/owlvit-base-patch32"
NAME = MODEL.split("/")[-1]  # short name for the output files, e.g. owlvit-base-patch32

detector = pipeline(model=MODEL, task="zero-shot-object-detection")

with open(RESULTS / f"predictions_{NAME}_{THRESHOLD}.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label", "score", "xmin", "ymin", "xmax", "ymax"])

    for path in paths:
        # makes a copy for drawing to leave original image intact
        # converting to RGB also lets grayscale and transparent images
        # work with the model and be saved as jpg
        image = Image.open(path).convert("RGB")
        # lower threshold makes the model report weaker guesses
        # any prediction with score below the threshold is discarded as likely
        # noise or false positives
        predictions = detector(
            image,
            candidate_labels=["rock", "crater"],
            threshold=THRESHOLD,
        )
        draw = ImageDraw.Draw(image)

        # box formatting
        line = max(3, image.width // 300)
        size = max(16, image.width // 60)
        font = ImageFont.load_default(size=size)

        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]
            xmin, ymin, xmax, ymax = box.values()

            # write prediction values to their respective columns in the csv file
            writer.writerow([path.name, label, round(score, 3), xmin, ymin, xmax, ymax])

            # visualize the predictions by drawing rectangles at the object's coordinates
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=line)
            # max keeps the label on the picture when a box touches the top edge
            draw.text(
                (xmin, max(0, ymin - size - 4)),
                f"{label}: {round(score, 2)}",
                fill="red",
                font=font,
            )

        image.save(RESULTS / f"{path.stem}_{NAME}_t{THRESHOLD}.jpg", quality=90)
        print(path.name, len(predictions), "detections")
