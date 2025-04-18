import torch, wandb
from torch.utils.data import DataLoader, TensorDataset
from .models import FlowMatchingNet
from .validate import validate
import argparse

def flow_loss(model, z0, z1, t):
    u = z1 - z0
    zt = t.unsqueeze(1)*z1 + (1-t).unsqueeze(1)*z0
    v = model(zt, t)
    return (v-u).pow(2).sum(1).mean()

def train(cfg):
    data = torch.load("dataset/cas9_embeddings.pt")
    embs = data["embeddings"]
    ds = TensorDataset(embs)
    dl = DataLoader(ds, cfg["flow"]["batch_size"], shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlowMatchingNet(cfg["flow"]["emb_dim"], cfg["flow"]["hidden_dim"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["flow"]["learning_rate"])

    best_val = float("inf"); patience=0
    best_model = None
    
    for epoch in range(cfg["flow"]["max_epochs"]):
        model.train()
        for batch, in dl:
            batch = batch.to(device)
            idx = torch.randperm(batch.size(0))
            z0, z1 = batch, batch[idx]
            t = torch.rand(batch.size(0), device=device)
            loss = flow_loss(model, z0, z1, t)
            opt.zero_grad(); loss.backward(); opt.step()
            wandb.log({"train/loss": loss.item(), "epoch": epoch})
        # validation stub (call validate.py)
        if epoch % cfg["flow"]["val_freq"]==0:
            val_loss = validate(model, cfg)  # implement in validate.py
            wandb.log({"val/loss": val_loss, "epoch": epoch})
            if val_loss < best_val:
                best_val=val_loss; patience=0
                torch.save(model.state_dict(), "best_flow.pt")
                best_model = model.state_dict().copy()
            else:
                patience+=1
                if patience>=cfg["flow"]["early_stop_patience"]:
                    break
    
    # Load best model before returning
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model

def parse_args():
    import yaml, sys
    cfg = yaml.safe_load(open("config.yaml"))
    wandb.init(entity=cfg["wandb"]["entity"],
               project=cfg["wandb"]["project"],
               config=cfg)
    return cfg

if __name__=="__main__":
    cfg=parse_args()
    train(cfg) 