import os
import torch
from torch.utils.data import Dataset
import pandas as pd

from utils import load_structure_from_cif, parse_ligand_from_cif

class FitnessDataset(Dataset):
    def __init__(self, csv_path, structure_dir, esm_model):
        self.data = pd.read_csv(csv_path)
        self.structure_dir = structure_dir
        self.tokenizers = esm_model.tokenizers
        self.structure_encoder = esm_model.get_structure_encoder()

    def __len__(self):
        return len(self.data)

    # def _get_structure_path(self, base_dir):
    #     for f in sorted(os.listdir(base_dir)):
    #         if f.startswith("fold_job") and f.endswith(".cif"):
    #             return os.path.join(base_dir, f)
    #         if f.startswith("wt") and f.endswith(".cif"):
    #             return os.path.join(base_dir, f)

    # def __getitem__(self):
    #     row = self.data
    #     mut_seq = row["mut_seq"]
    #     wt_seq = row["wt_seq"]
    
    def _get_structure_path(self, base_dir, prefix=None):
        for f in sorted(os.listdir(base_dir)):
            if prefix and not f.startswith(prefix):
                continue
            if f.endswith(".cif"):
                return os.path.join(base_dir, f)
        raise FileNotFoundError(f"{base_dir}에서 {prefix or ''}*.cif 파일을 찾지 못했습니다.")

    def __getitem__(self):
        row = self.data
        mut_seq = row["mut_seq"]
        wt_seq  = row["wt_seq"]
        # mut 구조
        try:
            mut_structure_path = self._get_structure_path(prefix="fold_job")
        except FileNotFoundError:
            raise FileNotFoundError(f"Mutant structure file not found in {self.structure_dir}")

        mut_protein_coords = load_structure_from_cif(mut_structure_path)
        mut_ligand_coords = parse_ligand_from_cif(mut_structure_path)
        mut_ligand_coords = mut_ligand_coords.unsqueeze(1).expand(-1, 3, -1)
        mut_full_coords = torch.cat([mut_protein_coords, mut_ligand_coords], dim=0)

        # wt 구조
        try:
            wt_structure_path = self._get_structure_path(prefix="wt")
        except FileNotFoundError:
            raise FileNotFoundError(f"Wild-type structure file not found in {self.structure_dir}")

        wt_protein_coords = load_structure_from_cif(wt_structure_path)
        wt_ligand_coords = parse_ligand_from_cif(wt_structure_path)
        wt_ligand_coords = wt_ligand_coords.unsqueeze(1).expand(-1, 3, -1)
        wt_full_coords = torch.cat([wt_protein_coords, wt_ligand_coords], dim=0)

        # sequence tokens
        mut_seq_tok = self.tokenizers.sequence(mut_seq, add_special_tokens=True)["input_ids"]
        mut_seq_tok = torch.tensor(mut_seq_tok, dtype=torch.long)
        wt_seq_tok = self.tokenizers.sequence(wt_seq, add_special_tokens=True)["input_ids"]
        wt_seq_tok = torch.tensor(wt_seq_tok, dtype=torch.long)

        # 패딩 처리 (ligand 부분 MASK로)
        def pad_sequence(seq_tok, target_len):
            pad_len = target_len - seq_tok.size(0)
            if pad_len > 0:
                pad_tok = torch.full((pad_len,), self.tokenizers.sequence.mask_token_id, dtype=torch.long)
                return torch.cat([seq_tok, pad_tok], dim=0)
            return seq_tok

        mut_seq_tok = pad_sequence(mut_seq_tok, mut_full_coords.size(0))
        wt_seq_tok = pad_sequence(wt_seq_tok, wt_full_coords.size(0))

        return {
            "mut_sequence_tokens": mut_seq_tok,
            "mut_structure_coords": mut_full_coords,
            "wt_sequence_tokens": wt_seq_tok,
            "wt_structure_coords": wt_full_coords
        }
