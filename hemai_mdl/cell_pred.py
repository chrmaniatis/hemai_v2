import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from collections import defaultdict
import time
from statistics import median

def write_folder_summary(folder_path: Path, label_counts: dict, total_images: int):
    rows = [
        {
            "label":       label_map[str(lbl)],
            "count":       cnt,
            "frequency_%": round(100 * cnt / total_images, 2) if total_images else 0,
        }
        for lbl, cnt in sorted(label_counts.items())
    ]
    rows.append({
        "label":       "TOTAL",
        "count":       total_images,
        "frequency_%": 100.00,
    })

    summary_df = pd.DataFrame(rows)
    csv_path   = folder_path / f"{folder_path.name}_label_frequency_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"\n✓ Summary saved → {csv_path}")
    print(summary_df.to_string(index=False))
    return csv_path

torch.backends.cudnn.benchmark = False

class MedVariableCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=1, padding=1)
        self.in1   = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.in2   = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.in3   = nn.InstanceNorm2d(128, affine=True)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.in4   = nn.InstanceNorm2d(256, affine=True)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = F.relu(self.in3(self.conv3(x)))
        x = F.relu(self.in4(self.conv4(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)
        
Root_Folder = "/content/drive/MyDrive/"
NUM_CLASSES   = 20

IMG_DIR       = Path(Root_Folder) / "Aima_pipeline/Images/cropped"
MODEL_PATH    = Path(Root_Folder) / "Aima_pipeline/best_model_weighted_entropy_rob.pkl"


start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


with open(MODEL_PATH, "rb") as f:
    best_model_data = pickle.load(f)

print(best_model_data.keys())


best_model_state = best_model_data["model_state"]
model = MedVariableCNN(NUM_CLASSES).to(device)
model.load_state_dict(best_model_state)
model.eval()
print("Model loaded successfully.")


transform = transforms.Compose([
    transforms.ToTensor()
])

label_map = {'0':'Basophils','1':'Blasts','2':'Eosinophils','3':'Erythroblasts','4':'Igs','5':'Lymphocytes','6':'Metamyelocytes',
       '7': 'Monocytes','8':'Neutrophils','9':'Echinocytes','10':'Elliptocytes','11':'Hypochromic','12':'Normal_cells',
       '13':'Ovalocytes','14':'Schistocytes','15':'Spherocytes','16':'Stomatocytes','17':'Target_cells','18':'Teardrops','19':'Plt'}

# Recursively collect all JPGs across all subfolders
jpg_files = sorted(IMG_DIR.rglob("*.jpg")) + sorted(IMG_DIR.rglob("*.JPG"))
jpg_files = [x for x in jpg_files]#[x for x in jpg_files if str(x).split("/")[-2].startswith("Can") or str(x).split("/")[-2].startswith("Fel")]

if not jpg_files:
    raise FileNotFoundError(f"No JPG images found under {IMG_DIR}")

folders: dict[Path, list[Path]] = defaultdict(list)
for p in jpg_files:
    folders[p.parent].append(p)


print(f"Found {len(jpg_files)} images across {len(folders)} folder(s). Starting inference...\n")

with torch.no_grad():
    for folder, images in sorted(folders.items()):
        print(f"\n{'─'*60}")
        print(f"  Processing folder: {folder.name}  ({len(images)} images)")
        print(f"{'─'*60}")

        areas = []

        for img_path in images:
            with Image.open(img_path) as img:
                w, h = img.size
                areas.append(w * h)

        median_area = median(areas)
        platelet_threshold = 0.6 * median_area

        label_counts: dict[int, int] = {}   # reset per folder

        for img_path in images:
              # Read only dimensions first
              with Image.open(img_path) as img:
                  w, h = img.size
                  area = w * h
                  aspect_ratio = min(w,h) / max(w,h)

                  if area < platelet_threshold and aspect_ratio > 0.65:
                      predicted_label = 19
                      predicted_name = "Plt"

                  else:
                      image = img.convert("RGB")
                      tensor = transform(image).unsqueeze(0).to(device)

                      logits = model(tensor)
                      predicted_label = int(logits.argmax(dim=1).item())
                      predicted_name = label_map[str(predicted_label)]


              new_name = f"{img_path.stem}_{predicted_name}{img_path.suffix}"
              new_path = img_path.parent / new_name
              img_path.rename(new_path)

              label_counts[predicted_label] = label_counts.get(predicted_label, 0) + 1

              print(f"{img_path.name} → {new_name}")

        # Write one CSV per folder, after all images in it are processed
        write_folder_summary(folder, label_counts, total_images=len(images))

print(f"\n {len(jpg_files)} images classified across {len(folders)} folder(s).")
elapsed = time.time() - start
hours, rem = divmod(elapsed, 3600)
mins, secs = divmod(rem, 60)
print(f"✅ Script finished in {int(hours):02d}h {int(mins):02d}m {secs:.2f}s")
