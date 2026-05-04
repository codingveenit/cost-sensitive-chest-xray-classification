import os
import argparse
import gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import densenet201, DenseNet201_Weights

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score

# ==============================
# ARGUMENTS
# ==============================

parser = argparse.ArgumentParser()

parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, default="../checkpoint")

args = parser.parse_args()

DATA_PATH = args.data_path
IMG_DIR = os.path.join(DATA_PATH, "images")
TRAIN_CSV = os.path.join(DATA_PATH, "train.csv")

CHECKPOINT_DIR = args.output_path
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
THRESH_PATH = os.path.join(CHECKPOINT_DIR, "best_thresholds.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# PARAMETERS
# ==============================

NUM_CLASSES = 20
BATCH_SIZE = 64
FOLDS = 4
IMAGE_SIZE = 320
EPOCHS = 8

# ==============================
# LOAD DATA
# ==============================

train_df = pd.read_csv(TRAIN_CSV)
label_cols = train_df.columns[1:]
labels = np.argmax(train_df[label_cols].values, axis=1)

# ==============================
# DATASET
# ==============================

class XRayDataset(Dataset):

    def __init__(self, df, transform=None):
        self.ids = df["id"].values
        self.labels = np.argmax(df[label_cols].values, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):

        img_id = self.ids[idx]
        img_path = os.path.join(IMG_DIR, img_id)

        if not os.path.exists(img_path):
            if os.path.exists(img_path + ".png"):
                img_path += ".png"
            elif os.path.exists(img_path + ".jpg"):
                img_path += ".jpg"

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

# ==============================
# TRANSFORMS
# ==============================

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==============================
# MODEL
# ==============================

def build_model():

    model = densenet201(weights=DenseNet201_Weights.DEFAULT)

    num_ftrs = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, NUM_CLASSES)
    )

    return model.to(DEVICE)

# ==============================
# TRAIN
# ==============================

def train_one_epoch(model, loader, optimizer, criterion):

    model.train()

    for imgs, labels in tqdm(loader):

        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(imgs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

# ==============================
# VALIDATION (returns probs)
# ==============================

def validate(model, loader):

    model.eval()

    preds_all = []
    targets_all = []
    probs_all = []

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(DEVICE)

            outputs = model(imgs)

            probs = torch.softmax(outputs,dim=1)

            preds = torch.argmax(probs,1).cpu().numpy()

            preds_all.extend(preds)
            targets_all.extend(labels.numpy())
            probs_all.append(probs.cpu().numpy())

    preds_all = np.array(preds_all)
    targets_all = np.array(targets_all)
    probs_all = np.concatenate(probs_all)

    acc = np.mean(preds_all==targets_all)
    f1 = f1_score(targets_all,preds_all,average="macro")
    recall = recall_score(targets_all,preds_all,average="macro")

    return acc,f1,recall,probs_all,targets_all

# ==============================
# ASYMMETRIC SCORE
# ==============================

def asymmetric_score(y_true,y_pred):

    scores=[]

    for c in range(NUM_CLASSES):

        tp=np.sum((y_true==c)&(y_pred==c))
        fp=np.sum((y_true!=c)&(y_pred==c))
        fn=np.sum((y_true==c)&(y_pred!=c))

        nc=np.sum(y_true==c)

        if nc==0:
            continue

        score=(tp-fp-5*fn)/nc

        scores.append(score)

    return np.mean(scores)

# ==============================
# THRESHOLD SEARCH
# ==============================

def find_best_thresholds(probs,targets):

    thresholds=[]

    for c in range(NUM_CLASSES):

        best_t=0.5
        best_score=-1e9

        for t in np.linspace(0.01,0.95,40):

            pred=(probs[:,c]>t).astype(int)
            true=(targets==c).astype(int)

            tp=np.sum((pred==1)&(true==1))
            fp=np.sum((pred==1)&(true==0))
            fn=np.sum((pred==0)&(true==1))

            nc=np.sum(true)

            if nc==0:
                continue

            score=(tp-fp-5*fn)/nc

            if score>best_score:

                best_score=score
                best_t=t

        thresholds.append(best_t)

    return np.array(thresholds)

# ==============================
# K-FOLD TRAINING
# ==============================

kf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=42)

best_score=-1

all_val_probs=[]
all_val_targets=[]

for fold,(train_idx,val_idx) in enumerate(kf.split(train_df,labels)):

    print(f"\n===== FOLD {fold} =====")

    train_split=train_df.iloc[train_idx]
    val_split=train_df.iloc[val_idx]

    train_ds=XRayDataset(train_split,train_tf)
    val_ds=XRayDataset(val_split,val_tf)

    train_loader=DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    val_loader=DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)

    model=build_model()

    optimizer=optim.AdamW(model.parameters(),lr=3e-4)
    criterion=nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}")

        train_one_epoch(model,train_loader,optimizer,criterion)

        val_acc,val_f1,val_recall,probs,targets=validate(model,val_loader)

        print("Val Acc:",val_acc,"F1:",val_f1,"Recall:",val_recall)

        if val_f1>best_score:

            best_score=val_f1

            torch.save(model.state_dict(),BEST_MODEL_PATH)

            print("Best model saved")

    all_val_probs.append(probs)
    all_val_targets.append(targets)

    del model
    torch.cuda.empty_cache()
    gc.collect()

print("\nTraining Complete")

# ==============================
# THRESHOLD TUNING
# ==============================

print("Combining validation predictions...")

val_probs=np.concatenate(all_val_probs)
val_targets=np.concatenate(all_val_targets)

baseline_preds=np.argmax(val_probs,axis=1)

baseline_score=asymmetric_score(val_targets,baseline_preds)

print("Baseline score:",baseline_score)

print("Searching optimal thresholds...")

best_thresholds=find_best_thresholds(val_probs,val_targets)

threshold_preds=[]

for probs in val_probs:

    detected=np.where(probs>best_thresholds)[0]

    if len(detected)==0:
        pred=np.argmax(probs)
    else:
        pred=detected[np.argmax(probs[detected])]

    threshold_preds.append(pred)

threshold_preds=np.array(threshold_preds)

threshold_score=asymmetric_score(val_targets,threshold_preds)

print("Threshold tuned score:",threshold_score)

np.save(THRESH_PATH,best_thresholds)

print("Thresholds saved:",THRESH_PATH)

# ====================================================
# FINAL TRAINING ON FULL DATASET
# ====================================================

print("\n🚀 Training FINAL model on FULL dataset")

from torch import amp

PROGRESSIVE_SIZES = [224, 320, 384]
PROGRESSIVE_EPOCHS = [3, 4, 6]

target_cols = label_cols

# Build new model
model = build_model()

optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

total_epochs = sum(PROGRESSIVE_EPOCHS)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_epochs
)

scaler = amp.GradScaler()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)

# ====================================================
# PROGRESSIVE RESIZING TRAINING
# ====================================================

for stage, img_size in enumerate(PROGRESSIVE_SIZES):

    epochs = PROGRESSIVE_EPOCHS[stage]

    print(f"\nStage {stage+1} | Image Size {img_size}")

    train_tf_stage = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05,0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = XRayDataset(train_df, train_tf_stage)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Freeze backbone first stage
    if stage == 0:

        backbone = model.module.features if isinstance(model, nn.DataParallel) else model.features

        for p in backbone.parameters():
            p.requires_grad = False

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        # Unfreeze backbone
        if stage == 0 and epoch == 2:

            backbone = model.module.features if isinstance(model, nn.DataParallel) else model.features

            for p in backbone.parameters():
                p.requires_grad = True

            print("Backbone unfrozen")

        model.train()

        running_correct = 0
        running_total = 0

        loop = tqdm(train_loader, desc="Train")

        for imgs, lbls in loop:

            imgs = imgs.to(DEVICE, non_blocking=True)
            lbls = lbls.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast():

                outputs = model(imgs)
                loss = criterion(outputs, lbls)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

            running_correct += (preds == lbls).sum().item()
            running_total += lbls.size(0)

            loop.set_postfix(
                loss=loss.item(),
                acc=running_correct / running_total
            )

        scheduler.step()

# ====================================================
# SAVE FINAL MODEL
# ====================================================

model_to_save = model.module if isinstance(model, nn.DataParallel) else model

torch.save(model_to_save.state_dict(), BEST_MODEL_PATH)

print("\n FINAL MODEL SAVED:", BEST_MODEL_PATH)

del model
torch.cuda.empty_cache()
gc.collect()

print("\nTraining finished.")