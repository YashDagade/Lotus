import os
import sys
import time
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from .models import FlowMatchingNet
from .validate import validate

def flow_loss(model, z0, z1, t):
    """MSE flow‐matching loss between paired embeddings."""
    u = z1 - z0
    zt = t.view(-1,1,1)*z1 + (1-t).view(-1,1,1)*z0
    v  = model(zt, t)
    return (v - u).pow(2).mean()

def compute_grad_norm(model):
    """L2 norm of all gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item()**2
    return total_norm**0.5

def train(cfg):
    # 1) Load embeddings
    emb_path = cfg["train_embeddings_path"]
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings file not found at {emb_path}")
    data = torch.load(emb_path)
    embs = data["embeddings"] if isinstance(data, dict) else data

    # 2) DataLoader
    batch_size = cfg["flow"]["batch_size"]
    dl = DataLoader(TensorDataset(embs), batch_size, shuffle=True)

    # 3) Model, optimizer, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowMatchingNet(cfg["flow"]["emb_dim"],
                            cfg["flow"]["hidden_dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["flow"]["learning_rate"])

    # 4) Training loop
    best_val    = float("inf")
    patience    = 0
    best_weights= None
    model_dir   = cfg["dirs"]["models"]
    os.makedirs(model_dir, exist_ok=True)
    best_path   = cfg["best_model_path"]

    global_step = 0
    for epoch in range(1, cfg["flow"]["max_epochs"]+1):
        epoch_start = time.time()
        model.train()
        batch_losses = []
        batch_grad_norms = []

        for (batch_embeds,) in dl:
            batch_embeds = batch_embeds.to(device)
            idx = torch.randperm(batch_embeds.size(0))
            z0, z1 = batch_embeds, batch_embeds[idx]
            t = torch.rand(batch_embeds.size(0), device=device)

            loss = flow_loss(model, z0, z1, t)
            optimizer.zero_grad()
            loss.backward()

            # compute grad norm BEFORE the step
            grad_norm = compute_grad_norm(model)
            batch_grad_norms.append(grad_norm)

            optimizer.step()

            batch_losses.append(loss.item())

            # log batch‐level metrics
            wandb.log({
                "batch/loss": loss.item(),
                "batch/grad_norm": grad_norm,
            }, step=global_step)
            global_step += 1

        # epoch metrics
        avg_loss = sum(batch_losses) / len(batch_losses)
        avg_grad = sum(batch_grad_norms) / len(batch_grad_norms)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # log epoch‐level metrics
        log_dict = {
            "train/loss": avg_loss,
            "train/lr": current_lr,
            "train/grad_norm": avg_grad,
            "train/epoch_time": epoch_time,
            "train/patience": patience,
            "epoch": epoch,
        }
        wandb.log(log_dict, step=epoch)

        # weight histograms
        for name, param in model.named_parameters():
            wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu().numpy())},
                      step=epoch)

        # validation + early stopping
        if epoch % cfg["flow"]["val_freq"] == 0:
            val_loss = validate(model, cfg)
            print(f"[Epoch {epoch}] val_loss = {val_loss:.4e}")
            wandb.log({"val/loss": val_loss, "epoch": epoch}, step=epoch)

            if val_loss < best_val:
                best_val = val_loss
                patience = 0
                best_weights = model.state_dict()
                print(f" ↳ New best model, saving to {best_path}")
                torch.save(best_weights, best_path)
            else:
                patience += 1
                if patience >= cfg["flow"]["early_stop_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # load best and return
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print("Loaded best model weights for return")

    return model

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open("config.yaml"))
    wandb.init(
        entity=cfg["wandb"]["entity"],
        project=cfg["wandb"]["project"],
        config=cfg
    )
    _ = train(cfg)
    print("Training finished successfully")
