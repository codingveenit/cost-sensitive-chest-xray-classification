import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import densenet201

torch.backends.cudnn.benchmark = True

# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--output", type=str, default="submission.csv")

args = parser.parse_args()

DATA_PATH = args.data_path
MODEL_PATH = args.model_path
OUTPUT_FILE = args.output

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# PARAMETERS
# ==============================

NUM_CLASSES = 20
BATCH_SIZE = 64
IMG_SIZE = 384
TEMPERATURE = 1.4
MARGIN_THRESHOLD = 0.08

print("Device:", DEVICE)

# ==============================
# LOAD TEST IDS
# ==============================

image_ids = sorted([
    f for f in os.listdir(DATA_PATH)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

test_df = pd.DataFrame({
    "id": image_ids
})

# sample_sub = pd.read_csv(os.path.join(DATA_PATH, "../sample_submission.csv"))

label_cols = [
"Atelectasis",
"Cardiomegaly",
"Consolidation",
"Edema",
"Effusion",
"Emphysema",
"Fibrosis",
"Hernia",
"Infiltration",
"Mass",
"Nodule",
"Pleural_Thickening",
"Pneumonia",
"Pneumothorax",
"Pneumoperitoneum",
"Pneumomediastinum",
"Subcutaneous Emphysema",
"Tortuous Aorta",
"Calcification of the Aorta",
"No Finding"
]

# ==============================
# LOAD THRESHOLDS
# ==============================

THRESH_PATH = os.path.join(os.path.dirname(MODEL_PATH), "best_thresholds.npy")
thresholds = np.load(THRESH_PATH)

print("Loaded thresholds:", thresholds)

# ==============================
# TEST DATASET
# ==============================

class TestDataset(Dataset):

    def __init__(self, df, img_dir, transform=None):
        self.ids = df["id"].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, img_id)

        if not os.path.exists(img_path):

            if os.path.exists(img_path + ".jpg"):
                img_path += ".jpg"

            elif os.path.exists(img_path + ".png"):
                img_path += ".png"

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, img_id

# ==============================
# TTA TRANSFORMS
# ==============================

normalize = transforms.Normalize(
    [0.485,0.456,0.406],
    [0.229,0.224,0.225]
)

tta_transforms = [

    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        normalize
    ]),

    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize
    ])
]

print("TTA passes:", len(tta_transforms))

# ==============================
# MODEL
# ==============================

def build_model():

    model = densenet201(weights=None)

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    return model

# ==============================
# LOAD MODEL
# ==============================

print("Loading DenseNet model...")

model = build_model()

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

model.load_state_dict(state_dict)

model = model.to(DEVICE)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.eval()

print("Model loaded successfully")

# ==============================
# TTA INFERENCE
# ==============================

def run_tta_inference():

    num_samples = len(test_df)

    tta_logits = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)

    ids_collector = []

    for i, tta_tf in enumerate(tta_transforms):

        print(f"TTA Pass {i+1}/{len(tta_transforms)}")

        dataset = TestDataset(test_df, DATA_PATH, tta_tf)

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        start_idx = 0

        with torch.no_grad():

            for imgs, ids in tqdm(loader):

                imgs = imgs.to(DEVICE)

                with torch.amp.autocast("cuda", enabled=(DEVICE.type=="cuda")):
                    outputs = model(imgs)

                logits = outputs.detach().cpu().numpy()

                bs = logits.shape[0]

                tta_logits[start_idx:start_idx+bs] += logits

                start_idx += bs

                if i == 0:
                    ids_collector.extend(ids)

    tta_logits /= len(tta_transforms)

    return tta_logits, ids_collector

# ==============================
# RUN INFERENCE
# ==============================

logits, ids = run_tta_inference()

# ==============================
# SOFTMAX + TEMPERATURE
# ==============================

scaled_logits = logits / TEMPERATURE

probs = torch.softmax(
    torch.from_numpy(scaled_logits),
    dim=1
).numpy()

probs = np.clip(probs,1e-6,1-1e-6)

# ==============================
# APPLY THRESHOLD + MARGIN RULE
# ==============================

predictions = []

for p in probs:

    detected = np.where(p > thresholds)[0]

    if len(detected) == 0:
        pred = np.argmax(p)
    else:
        pred = detected[np.argmax(p[detected])]

    # margin rule
    top2 = np.argsort(p)[-2:]
    margin = p[top2[-1]] - p[top2[-2]]

    if margin < MARGIN_THRESHOLD:
        pred = np.argmax(p)

    predictions.append(pred)

predictions = np.array(predictions)

print("Prediction generation complete.")

# ==============================
# CREATE SUBMISSION
# ==============================

submission = pd.DataFrame()

submission["id"] = ids

for i, col in enumerate(label_cols):
    submission[col] = (predictions == i).astype(int)

submission.to_csv(OUTPUT_FILE, index=False)

print("Submission saved:", OUTPUT_FILE)