# infer_fitness.py
import torch, pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import FitnessDataset
from model import ESM3WithFitnessHead
from utils import spearman_corr, custom_collate_fn
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.pretrained import load_local_model


@torch.no_grad()
def run(csv_path, struct_dir, ckpt, out_csv, device="cuda:0", bs=2, d_model=1536):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    esm3 = load_local_model(ESM3_OPEN_SMALL, device=dev)
    model = ESM3WithFitnessHead(esm3, d_model=d_model).to(dev)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    ds = FitnessDataset(csv_path, struct_dir, esm3)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=custom_collate_fn)

    preds_all = []
    for batch in tqdm(dl, desc="Inference"):
        inputs = {k: v.to(dev) for k, v in batch.items() if k != "fitness_label"}
        preds = model(**inputs)
        preds = preds.squeeze(-1) if preds.ndim == 2 and preds.size(-1) == 1 else preds
        preds_all.append(preds.detach().cpu())

    preds_all = torch.cat(preds_all).numpy()  # 순서 = CSV 순서
    df = pd.read_csv(csv_path)
    df["prediction"] = preds_all[:len(df)]
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")
    
    return out_csv