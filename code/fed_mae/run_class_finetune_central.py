"""
Centralized fine-tuning for Retina dataset (ViT backbone, MAE-pretrained)
- 1-logit head + BCE/Focal (configurable)
- Robust pos_embed resize
- CLS-token pooling if model outputs per-token logits
- Inference-time TTA (hflip, vflip, transpose variants; averages LOGITS)
- Two LR param groups (head vs encoder)
- Early-friendly defaults; edit in Args below
"""

import os, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

# Make fed_mae importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import fed_mae.models_vit as models_vit
except ImportError:
    print("Error: Could not import 'fed_mae.models_vit'. Check PYTHONPATH.")
    sys.exit(1)


# =========================
# Args (edit here)
# =========================
class Args:
    # Paths
    finetune   = "/cta/users/undergrad2/SSL-FL/out_pretrain_retina2_35K/checkpoint-1599.pth"
    data_path  = "/cta/users/undergrad2/SSL-FL/data/Retina"
    output_dir = "/cta/users/undergrad2/SSL-FL/out_finetune_central_tta_focal"

    # Model
    model        = "vit_base_patch16"
    nb_classes   = 2           # API compat; we override head to 1-logit
    global_pool  = True
    drop_path    = 0.05        # keep modest in fine-tuning
    unfreeze_last= 8           # try 8; you can bump to 12 if stable

    # Training
    epochs         = 300
    batch_size     = 256
    weight_decay   = 0.08      # a touch higher than 0.05 to tame overfit
    base_head_lr   = 2e-3      # head learns faster
    base_enc_lr    = 5e-4      # encoder adapts gently
    warmup_epochs  = 5         # fine-tune: short warmup works better

    # Loss selection
    USE_FOCAL      = True      # True: Focal; False: BCEWithLogits
    focal_gamma    = 2.0
    # alpha for positive class; if None we compute from class counts (pos = n_pos/(n_pos+n_neg))
    focal_alpha_pos = None

    # Sampling
    USE_SAMPLER    = False     # default False to avoid double-balancing with focal/BCE weighting

    # Inference-time augmentation (TTA)
    USE_TTA        = True      # enable TTA at validation

    # Device
    device = "cuda"


# =========================
# Dataset
# =========================
class RetinaCentralDataset(Dataset):
    def __init__(self, csv_path, img_root, train=True, img_size=224):
        self.df = pd.read_csv(csv_path)
        # assumes: col0=relative path, col1=label
        self.paths  = self.df.iloc[:, 0].astype(str).tolist()
        self.labels = self.df.iloc[:, 1].astype(int).tolist()
        self.img_root = img_root

        if train:
            self.tfm = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.paths[idx])
        img = Image.open(img_path).convert("RGB")
        return self.tfm(img), self.labels[idx]


# =========================
# Losses
# =========================
class SigmoidFocalLossWithLogits(nn.Module):
    """
    Binary sigmoid focal loss with logits (stable), per Lin et al. 2017.
    alpha_pos: weight for positive class (float in [0,1]); alpha_neg = 1 - alpha_pos
    gamma: focusing parameter >= 0
    """
    def __init__(self, alpha_pos: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = 1.0 - float(alpha_pos)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        # logits: [B], targets: [B] float in {0,1}
        # BCE with logits (no reduction), then focal modulating factor
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # p_t = sigmoid(logit) if y=1, else 1 - sigmoid(logit)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        # alpha_t
        alpha_t = self.alpha_pos * targets + self.alpha_neg * (1 - targets)
        # focal
        mod = torch.pow(1 - p_t, self.gamma)
        loss = alpha_t * mod * bce
        return loss.mean()


# =========================
# Train / Validate
# =========================
def _cls_pool_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Ensures we return [B] logits even if the model outputs per-token logits [B,T,1].
    Uses CLS token (index 0), which is standard for ViTs.
    """
    if logits.dim() == 3:       # [B, T, 1]
        logits = logits[:, 0, :]
    return logits.squeeze(1)     # [B]

def forward_with_tta(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    """
    Average LOGITS over simple TTA views (hflip, vflip, transpose).
    """
    views = [
        imgs,
        torch.flip(imgs, dims=[-1]),               # hflip
        torch.flip(imgs, dims=[-2]),               # vflip
        imgs.transpose(-1, -2),                    # transpose H<->W
        torch.flip(imgs.transpose(-1, -2), [-1]),  # transpose + hflip
        torch.flip(imgs.transpose(-1, -2), [-2]),  # transpose + vflip
    ]
    outs = []
    for v in views:
        o = model(v)
        outs.append(_cls_pool_logits(o))           # [B]
    return torch.stack(outs, dim=0).mean(dim=0)    # [B]

def train_one_epoch_simple(model, loader, optimizer, device, epoch, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for it, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        logits = _cls_pool_logits(logits)              # [B]
        loss   = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        with torch.no_grad():
            probs    = torch.sigmoid(logits)
            pred_cls = (probs > 0.5).long()
            correct += (pred_cls == labels).sum().item()

        if (it + 1) % 50 == 0:
            acc = 100.0 * correct / max(1, total)
            avg_loss = running_loss / max(1, total)
            lr = optimizer.param_groups[0]["lr"]
            print(f"[Epoch {epoch} | {it+1}/{len(loader)}] acc={acc:.2f}% loss={avg_loss:.4f} lr={lr:.6f}", flush=True)

    epoch_loss = running_loss / max(1, total)
    epoch_acc  = 100.0 * correct / max(1, total)
    print(f"Epoch {epoch}: train loss={epoch_loss:.4f}  train acc={epoch_acc:.2f}%", flush=True)
    return {"loss": epoch_loss, "acc": epoch_acc}

@torch.no_grad()
def validate(model, loader, device, criterion=None, use_tta=True):
    model.eval()
    total_loss, n = 0.0, 0
    all_probs, all_targets = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_tta:
            logits = forward_with_tta(model, imgs)     # [B]
        else:
            logits = model(imgs)
            logits = _cls_pool_logits(logits)          # [B]

        if criterion is not None:
            loss = criterion(logits, labels.float())
            total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        probs = torch.sigmoid(logits).detach().cpu()
        all_probs.append(probs)
        all_targets.append(labels.detach().cpu())

    y_true = torch.cat(all_targets).numpy().astype(int)
    y_prob = torch.cat(all_probs).numpy().astype(float)
    val_loss = (total_loss / max(1, n)) if criterion is not None else float("nan")

    # Metrics
    try:
        val_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        val_auc = float("nan")

    ts = np.linspace(0.05, 0.95, 19)
    accs, f1s, baccs = [], [], []
    for t in ts:
        y_pred = (y_prob > t).astype(int)
        accs.append((y_pred == y_true).mean())
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        baccs.append(balanced_accuracy_score(y_true, y_pred))
    i_best = int(np.argmax(f1s))  # or np.argmax(baccs) to optimize bal acc
    best_t = float(ts[i_best])
    val_acc_best = float(accs[i_best]) * 100.0
    val_f1_best = float(f1s[i_best])
    val_bacc_best = float(baccs[i_best])

    return val_loss, val_acc_best, val_auc, best_t, val_f1_best, val_bacc_best


# =========================
# Main
# =========================
def main():
    args = Args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # ---- Build model ----
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool
    )
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)  # single logit
    print("? Installed 1-logit linear head")

    # ---- Load MAE checkpoint (with pos_embed adjust) ----
    if args.finetune and os.path.isfile(args.finetune):
        print(f"=> Loading MAE pretrain from {args.finetune}")
        ckpt = torch.load(args.finetune, map_location="cpu")
        state = ckpt.get("model", ckpt)
        enc_state = {k: v for k, v in state.items() if not k.startswith("decoder")}

        if "pos_embed" in enc_state and hasattr(model, "pos_embed"):
            pe_ckpt = enc_state["pos_embed"]       # [1, N_ckpt, C]
            pe_curr = model.pos_embed              # [1, N_curr, C]
            if pe_ckpt.shape != pe_curr.shape:
                num_patches_curr = getattr(model.patch_embed, "num_patches", pe_curr.shape[1]-1)
                num_extra_curr = pe_curr.shape[1] - num_patches_curr  # 0/1

                n_ckpt = pe_ckpt.shape[1]
                has_cls_ckpt = int((n_ckpt - 1) ** 0.5) ** 2 == (n_ckpt - 1)
                if has_cls_ckpt:
                    cls_ckpt, patch_ckpt = pe_ckpt[:, :1], pe_ckpt[:, 1:]
                else:
                    cls_ckpt, patch_ckpt = None, pe_ckpt

                C = patch_ckpt.shape[-1]
                hw_ckpt = int(patch_ckpt.shape[1] ** 0.5)
                patch_ckpt = patch_ckpt.reshape(1, hw_ckpt, hw_ckpt, C).permute(0,3,1,2)

                hw_curr = int(num_patches_curr ** 0.5)
                patch_resized = F.interpolate(patch_ckpt, size=(hw_curr, hw_curr),
                                              mode="bicubic", align_corners=False)
                patch_resized = patch_resized.permute(0,2,3,1).reshape(1, hw_curr*hw_curr, C)

                if num_extra_curr > 0:
                    if cls_ckpt is None:
                        cls_ckpt = pe_curr[:, :num_extra_curr]
                    pe_new = torch.cat([cls_ckpt[:, :num_extra_curr], patch_resized], dim=1)
                else:
                    pe_new = patch_resized

                enc_state["pos_embed"] = pe_new
                print(f"=> Adjusted pos_embed from {tuple(pe_ckpt.shape)} to {tuple(pe_new.shape)}")

        msg = model.load_state_dict(enc_state, strict=False)
        try:
            print(f"[finetune load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        except Exception:
            missing, unexpected = msg
            print(f"[finetune load] missing={len(missing)} unexpected={len(unexpected)}")

    # ---- Freeze -> unfreeze last K + head + norm ----
    print("Freezing layers...")
    for p in model.parameters(): p.requires_grad = False
    try:
        num_blocks = len(model.blocks)
    except AttributeError:
        num_blocks = 12
    unf_k = max(int(args.unfreeze_last), 0)
    print(f"Model has {num_blocks} blocks. Unfreezing the last {unf_k}.")
    for name, p in model.named_parameters():
        if (any(name.startswith(f"blocks.{i}") for i in range(num_blocks - unf_k, num_blocks))
            or name.startswith("head") or name.startswith("norm")):
            p.requires_grad = True

    t_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"=> Trainable params: {t_params/1e6:.2f}M / {all_params/1e6:.2f}M ({100*t_params/all_params:.2f}%)")
    model.to(device)

    # ---- Data ----
    train_csv = os.path.join(args.data_path, "train.csv")
    test_csv  = os.path.join(args.data_path, "test.csv")  # used as "val" here
    train_ds = RetinaCentralDataset(train_csv, os.path.join(args.data_path, "train"), train=True)
    val_ds   = RetinaCentralDataset(test_csv,  os.path.join(args.data_path, "test"),  train=False)

    # Labels for class stats
    try:
        labels_tensor = torch.as_tensor(train_ds.labels, dtype=torch.long)
    except Exception:
        tmp_loader = DataLoader(train_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)
        lbls = []
        for _, y in tmp_loader: lbls.append(y)
        labels_tensor = torch.cat(lbls).long()

    class_count = torch.bincount(labels_tensor)
    if len(class_count) < 2:
        class_count = torch.tensor([labels_tensor.eq(0).sum(), labels_tensor.eq(1).sum()])
    n_neg, n_pos = class_count[0].item(), class_count[1].item()
    print(f"class counts: neg={n_neg} pos={n_pos}")

    # ---- Dataloaders (sampler optional; default off) ----
    if Args.USE_SAMPLER:
        class_weight = (class_count.sum().float() / class_count.float())
        sample_weight = class_weight[labels_tensor].cpu().numpy()
        sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        print("=> Using WeightedRandomSampler (turn OFF loss alpha/pos_weight effects)")
    else:
        dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    dl_val   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Criterion (Focal or BCE) ----
    if Args.USE_FOCAL:
        if Args.focal_alpha_pos is None:
            # α_pos ≈ class prior of positive; with gamma>0 this usually works well
            alpha_pos = n_pos / max(1, (n_pos + n_neg))
        else:
            alpha_pos = float(Args.focal_alpha_pos)
        criterion = SigmoidFocalLossWithLogits(alpha_pos=alpha_pos, gamma=Args.focal_gamma)
        print(f"=> Using FocalLoss (alpha_pos={alpha_pos:.3f}, gamma={Args.focal_gamma})")
    else:
        # BCEWithLogits with pos_weight = n_neg/n_pos (only if no sampler)
        if Args.USE_SAMPLER:
            criterion = nn.BCEWithLogitsLoss()
            print("=> Using BCEWithLogitsLoss (no pos_weight; sampler handles balance)")
        else:
            pos_weight = torch.tensor([n_neg / max(1, n_pos)], device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"=> Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.3f}")

    # ---- Optimizer (two groups) ----
    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (head_params if n.startswith("head") else enc_params).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": args.base_head_lr},
         {"params": enc_params,  "lr": args.base_enc_lr}],
        weight_decay=args.weight_decay
    )
    print(f"=> AdamW LRs: head={args.base_head_lr:.3g}, encoder={args.base_enc_lr:.3g}, wd={args.weight_decay}")

    # ---- LR schedule: warmup -> cosine ----
    eta_min_head = args.base_head_lr * 0.1
    eta_min_enc  = args.base_enc_lr  * 0.1
    def set_epoch_lr(ep):
        if ep < args.warmup_epochs:
            f = float(ep + 1) / max(1, args.warmup_epochs)
            lr_h, lr_e = args.base_head_lr * f, args.base_enc_lr * f
        else:
            t = ep - args.warmup_epochs
            T = max(1, args.epochs - args.warmup_epochs)
            cos_t = 0.5 * (1.0 + math.cos(math.pi * t / T))
            lr_h = eta_min_head + (args.base_head_lr - eta_min_head) * cos_t
            lr_e = eta_min_enc  + (args.base_enc_lr  - eta_min_enc)  * cos_t
        optimizer.param_groups[0]["lr"] = lr_h
        optimizer.param_groups[1]["lr"] = lr_e
        return lr_h, lr_e

    # ---- Train ----
    best_score = -1.0
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        lr_h, lr_e = set_epoch_lr(epoch)
        print(f"\n===== Epoch {epoch+1} / {args.epochs} =====")
        print(f"epoch {epoch} LRs -> head={lr_h:.6g} enc={lr_e:.6g}")

        _ = train_one_epoch_simple(model, dl_train, optimizer, device, epoch, criterion)
        val_loss, val_acc_best, val_auc, best_t, val_f1, val_bacc = validate(
            model, dl_val, device, criterion, use_tta=Args.USE_TTA
        )
        print(f"Val: acc@bestT={val_acc_best:.2f}% (T={best_t:.2f})  AUC={val_auc:.3f}  F1={val_f1:.3f}  BalAcc={val_bacc:.3f}  loss={val_loss:.4f}")

        select_score = val_auc if not math.isnan(val_auc) else (val_acc_best / 100.0)
        if select_score > best_score:
            best_score = select_score
            torch.save(
                {"model": model.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1,
                 "best_score": best_score,
                 "best_threshold": best_t},
                os.path.join(args.output_dir, "checkpoint-best.pth")
            )
            print(f"? Saved new best checkpoint (score={best_score:.4f})")

    print(f"?? Training complete. Best score={best_score:.4f}")


if __name__ == "__main__":
    main()
