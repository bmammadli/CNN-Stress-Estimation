# Imports
import os
import numpy as np
import cv2
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Device set-up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.is_available()



# Preparing the Data

IMG_HEIGHT = 150
IMG_WIDTH = 780
IMG_CHS = 3

BATCH_SIZE = 32

train_dir = "A"
valid_dir = "B"

train_label_file = "stress_labels_a.txt"
valid_label_file = "stress_labels_b.txt"


# ------------------------------------------------------------
# Load raw stress labels and compute max values for each experiment
# ------------------------------------------------------------
train_labels_raw = np.loadtxt(train_label_file, dtype=np.float32)
valid_labels_raw = np.loadtxt(valid_label_file, dtype=np.float32)

max_stress_A = train_labels_raw.max()
max_stress_B = valid_labels_raw.max()

#print("Max stress A:", max_stress_A)
#print("Max stress B:", max_stress_B)


# ------------------------------------------------------------
# Compute image normalization statistics from training images only
# ------------------------------------------------------------
def compute_image_mean_std(image_dir):
    image_names = sorted([f for f in os.listdir(image_dir) if f.endswith(".bmp")])

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    total_pixels = 0

    for name in image_names:
        img_path = os.path.join(image_dir, name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        if img.shape != (IMG_HEIGHT, IMG_WIDTH):
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        img = img.astype(np.float32) / 255.0

        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        total_pixels += img.size

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean ** 2)

    return mean, std


image_mean, image_std = compute_image_mean_std(train_dir)

#print("Image mean:", image_mean)
#print("Image std:", image_std)

# ------------------------------------------------------------
# Save initial parametrs into CSV
# ------------------------------------------------------------
max_values_df = pd.DataFrame({
    "Parameters": ["A_Max", "B_Max","Img_Mean","Img_Std"],
    "values": [max_stress_A, max_stress_B, image_mean, image_std]
})
max_values_df.to_csv("initial_exp_parameters.csv", index=False)


# ------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, image_dir, label_file, max_stress, image_mean, image_std):
        self.image_dir = image_dir
        self.max_stress = max_stress
        self.image_mean = image_mean
        self.image_std = image_std

        # Read stress labels
        labels = np.loadtxt(label_file, dtype=np.float32)

        # Normalize by experiment maximum stress
        self.labels = labels / max_stress

        # Read image names in sorted order
        self.image_names = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".bmp")
        ])

        assert len(self.image_names) == len(self.labels), \
            f"Mismatch in {image_dir}: {len(self.image_names)} images vs {len(self.labels)} labels"

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Safety resize (only if needed)
        if img.shape != (IMG_HEIGHT, IMG_WIDTH):
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Scale to [0,1]
        img = img.astype(np.float32) / 255.0

        # Normalize using training image statistics
        img = (img - self.image_mean) / self.image_std

        # Convert grayscale → 3 channels for ResNet
        img = np.stack([img, img, img], axis=0)

        x = torch.tensor(img, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.image_names)


# ------------------------------------------------------------
# Create datasets
# ------------------------------------------------------------
train_data = MyDataset(
    train_dir, train_label_file,
    max_stress_A,
    image_mean, image_std
)

valid_data = MyDataset(
    valid_dir, valid_label_file,
    max_stress_B,
    image_mean, image_std
)


train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)


train_N = len(train_loader.dataset)
valid_N = len(valid_loader.dataset)

#print("Training samples:", train_N)
#print("Validation samples:", valid_N)


# Build the Model

class MyModel(nn.Module):
    def __init__(self, dropout_p=0.2):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.regressor(x)
        return x


base_model = MyModel(dropout_p=0.3).to(device)

# Disable torch.compile on this HPC cluster
model = base_model
print("Using normal model (torch.compile disabled)")

# Helper for compiled / non-compiled model
def get_core_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# Freeze / unfreeze helpers

def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model):
    # freeze everything first
    for param in model.backbone.parameters():
        param.requires_grad = False

    # unfreeze layer3-> decreased the accuracy
   # for param in model.backbone.layer3.parameters():
       # param.requires_grad = True

    # unfreeze layer4
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True

    # keep regressor trainable
    for param in model.regressor.parameters():
        param.requires_grad = True


# Loss function and optimizer (Phase 1)

loss_function = nn.HuberLoss()

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

#Data Augumentation   #did not show much improvement

#random_transforms = transforms.Compose([
#    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
#])

def train():
    loss = 0
    mae = 0

    model.train()

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        output = model(x).squeeze(1)
        # x_aug = random_transforms(x)
        # output = model(x_aug).squeeze(1)

        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()

        # Convert normalized predictions and labels back to MPa
        pred_stress = output * max_stress_A
        true_stress = y * max_stress_A

        loss += batch_loss.item()
        mae += torch.mean(torch.abs(pred_stress - true_stress)).item()

    loss /= len(train_loader)
    mae /= len(train_loader)

    print('Train - Loss: {:.4f} MAE: {:.4f} MPa'.format(loss, mae))

    return loss, mae

def validate():
    loss = 0
    mae = 0

    model.eval()

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x).squeeze(1)

            batch_loss = loss_function(output, y)

            # Convert normalized predictions and labels back to MPa
            pred_stress = output * max_stress_B
            true_stress = y * max_stress_B

            loss += batch_loss.item()
            mae += torch.mean(torch.abs(pred_stress - true_stress)).item()

    loss /= len(valid_loader)
    mae /= len(valid_loader)

    print('Valid - Loss: {:.4f} MAE: {:.4f} MPa'.format(loss, mae))

    return loss, mae

head_epochs = 5
head_log = "training_head.csv"

with open(head_log, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_mae", "valid_loss", "valid_mae"])

best_valid_mae = float("inf")

freeze_backbone(model)

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

for epoch in range(head_epochs):
    print(f"\nPhase 1 - Epoch {epoch+1}/{head_epochs}")

    train_loss, train_mae = train()
    valid_loss, valid_mae = validate()

    with open(head_log, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_mae, valid_loss, valid_mae])

# Phase 2 - Fine-tune only the last ResNet block

finetune_epochs = 20
finetune_log = "training_finetune.csv"
patience = 4
epochs_without_improvement = 0

with open(finetune_log, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "train_mae", "valid_loss", "valid_mae"])

unfreeze_last_blocks(model)

optimizer = Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=5e-6,
    weight_decay=1e-4
)

for epoch in range(finetune_epochs):
    print(f"\nPhase 2 - Epoch {epoch+1}/{finetune_epochs}")

    train_loss, train_mae = train()
    valid_loss, valid_mae = validate()

    with open(finetune_log, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss, train_mae, valid_loss, valid_mae])

    if valid_mae < best_valid_mae:
        best_valid_mae = valid_mae
        epochs_without_improvement = 0
        torch.save(get_core_model(model).state_dict(), "stainless_steel_resnet18_best.pth")
        print("Best model saved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement.")
        break

# ============================================================
# Evaluate best fine-tuned model, save CSV files, and plot figures
# ============================================================


# ------------------------------------------------------------
# 1. Load best saved fine-tuned model
# ------------------------------------------------------------
get_core_model(model).load_state_dict(
    torch.load("stainless_steel_resnet18_best.pth", map_location=device)
)
model.eval()


# ------------------------------------------------------------
# 2. Create evaluation loaders (shuffle=False keeps correct order)
# ------------------------------------------------------------
train_eval_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
valid_eval_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)


# ------------------------------------------------------------
# 3. Helper: get predictions and targets in MPa
#    (model predicts normalized stress = stress / max_stress)
# ------------------------------------------------------------
def get_predictions_and_targets(loader, max_stress):
    preds_all = []
    targets_all = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(device)
            y = y.to(device)

            output = model(x).squeeze(1)

            # Convert normalized values back to MPa
            pred_stress = output * max_stress
            true_stress = y * max_stress

            preds_all.extend(pred_stress.cpu().numpy())
            targets_all.extend(true_stress.cpu().numpy())

    return np.array(targets_all), np.array(preds_all)


# ------------------------------------------------------------
# 4. Helper: compute metrics
# ------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return r2, rmse, mae


# ------------------------------------------------------------
# 5. Get predictions
# ------------------------------------------------------------
train_true, train_pred = get_predictions_and_targets(train_eval_loader, max_stress_A)
valid_true, valid_pred = get_predictions_and_targets(valid_eval_loader, max_stress_B)

train_r2, train_rmse, train_mae = compute_metrics(train_true, train_pred)
valid_r2, valid_rmse, valid_mae = compute_metrics(valid_true, valid_pred)

print("Training metrics:")
print(f"R²   = {train_r2:.4f}")
print(f"RMSE = {train_rmse:.4f} MPa")
print(f"MAE  = {train_mae:.4f} MPa")

print("\nValidation metrics:")
print(f"R²   = {valid_r2:.4f}")
print(f"RMSE = {valid_rmse:.4f} MPa")
print(f"MAE  = {valid_mae:.4f} MPa")


# ------------------------------------------------------------
# 6. Save actual vs predicted values into CSV files
# ------------------------------------------------------------
train_pred_df = pd.DataFrame({
    "frame_index": np.arange(len(train_true)),
    "actual_stress_mpa": train_true,
    "predicted_stress_mpa": train_pred,
    "actual_reduction_factor": train_true / max_stress_A,
    "predicted_reduction_factor": train_pred / max_stress_A
})
train_pred_df.to_csv("train_predictions_A.csv", index=False)

valid_pred_df = pd.DataFrame({
    "frame_index": np.arange(len(valid_true)),
    "actual_stress_mpa": valid_true,
    "predicted_stress_mpa": valid_pred,
    "actual_reduction_factor": valid_true / max_stress_B,
    "predicted_reduction_factor": valid_pred / max_stress_B
})
valid_pred_df.to_csv("valid_predictions_B.csv", index=False)


# ------------------------------------------------------------
# 7. Save metrics into CSV file
# ------------------------------------------------------------
metrics_df = pd.DataFrame({
    "dataset": ["train_A", "valid_B"],
    "max_stress_mpa": [max_stress_A, max_stress_B],
    "R2": [train_r2, valid_r2],
    "RMSE_MPa": [train_rmse, valid_rmse],
    "MAE_MPa": [train_mae, valid_mae]
})
metrics_df.to_csv("metrics_train_valid.csv", index=False)


# ------------------------------------------------------------
# 8. Global plot style for readability
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16
})


# ------------------------------------------------------------
# 9. Scatter plot function
# ------------------------------------------------------------
def plot_scatter(y_true, y_pred, title, output_name, legend_label):

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        y_true,
        y_pred,
        s=60,
        alpha=0.85,
        label=legend_label
    )

    #min_val = min(y_true.min(), y_pred.min())
    #max_val = max(y_true.max(), y_pred.max())

    # New fixed limits
    ax.set_xlim(500, 750)
    ax.set_ylim(500, 750)

    ax.plot(
        #[min_val, max_val],
        #[min_val, max_val],
        [500, 750],
        [500, 750],
        'r--',
        linewidth=2,
        label="Perfect fit"
    )

    ax.set_xlabel("Actual Stress (MPa)")
    ax.set_ylabel("Predicted Stress (MPa)")
    ax.set_title(title)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_name, dpi=600, bbox_inches="tight")
    # plt.show()


# ------------------------------------------------------------
# 10. Plot training scatter
# ------------------------------------------------------------
plot_scatter(
    train_true,
    train_pred,
    title="Training Data",
    output_name="scatter_training_A.png",
    legend_label="Predictions training data"
)


# ------------------------------------------------------------
# 11. Plot validation scatter
# ------------------------------------------------------------
plot_scatter(
    valid_true,
    valid_pred,
    title="Validation Data",
    output_name="scatter_validation_B.png",
    legend_label="Predictions validation data"
)


# ------------------------------------------------------------
# 12. Plot metrics in 3 side-by-side horizontal bar charts
#     R² in %, RMSE in MPa, MAE in MPa
# ------------------------------------------------------------
dataset_names = ["Training data", "Validation data"]

r2_vals = [train_r2 * 100, valid_r2 * 100]
rmse_vals = [train_rmse, valid_rmse]
mae_vals = [train_mae, valid_mae]

y = np.arange(len(dataset_names))

plt.rcParams.update({
    "font.family": "Arial",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)

# -----------------------------
# R² subplot
# -----------------------------
axes[0].barh(y, r2_vals, color="skyblue")
#axes[0].set_title("R²")
axes[0].set_xlabel("R² (%)")
axes[0].set_yticks(y)
axes[0].set_yticklabels(dataset_names)
axes[0].grid(True, axis="x", linestyle="--", alpha=0.5)

for i, v in enumerate(r2_vals):
    axes[0].text(v, i, f" {v:.1f}%", va="center", fontsize=12)

# -----------------------------
# RMSE subplot
# -----------------------------
axes[1].barh(y, rmse_vals, color="lightgreen")
#axes[1].set_title("RMSE")
axes[1].set_xlabel("RMSE (MPa)")
axes[1].grid(True, axis="x", linestyle="--", alpha=0.5)

for i, v in enumerate(rmse_vals):
    axes[1].text(v, i, f" {v:.2f}", va="center", fontsize=12)

# -----------------------------
# MAE subplot
# -----------------------------
axes[2].barh(y, mae_vals, color="salmon")
#axes[2].set_title("MAE")
axes[2].set_xlabel("MAE (MPa)")
axes[2].grid(True, axis="x", linestyle="--", alpha=0.5)

for i, v in enumerate(mae_vals):
    axes[2].text(v, i, f" {v:.2f}", va="center", fontsize=12)

# Put Training data on top
axes[0].invert_yaxis()

plt.tight_layout()
plt.savefig("metrics_train_valid_3panel.png", dpi=600, bbox_inches="tight")
# plt.show()
