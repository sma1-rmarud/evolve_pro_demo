# utils.py
import torch
import torch.nn as nn
import numpy as np
from Bio.PDB.MMCIFParser import MMCIFParser
import numpy as np

def load_structure_from_cif(cif_path: str) -> torch.Tensor:
    """
    Parses .cif file to extract backbone coordinates (N, CA, C) in order.
    Returns tensor of shape (L, 3, 3) where 3 atoms per residue.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)

    coords = []
    current_residue = None
    current_atoms = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                atom_coords = []
                for atom_name in ["N", "CA", "C"]:
                    if atom_name in residue:
                        atom_coords.append(residue[atom_name].get_coord())
                if len(atom_coords) == 3:
                    coords.append(atom_coords)
    coords = np.array(coords)  # (L, 3, 3)
    return torch.tensor(coords, dtype=torch.float32)


def parse_ligand_from_cif(cif_path: str, exclude_resnames={"HOH", "WAT"}) -> torch.Tensor:
    """
    Parses ligand atoms from .cif file (lines starting with HETATM and not water).
    Returns (L_ligand, 3) tensor of coordinates.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("ligand", cif_path)

    ligand_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                hetfield, resname, _ = residue.id
                if hetfield != " " and resname not in exclude_resnames:
                    for atom in residue:
                        ligand_coords.append(atom.get_coord())

    if len(ligand_coords) == 0:
        raise ValueError(f"No ligand atoms found in {cif_path}")

    coords = np.array(ligand_coords)  # (L_ligand, 3)
    return torch.tensor(coords, dtype=torch.float32)

def custom_collate_fn(batch):
    keys = batch[0].keys()
    batch_out = {}
    pad_token_id = 0
    
    for key in keys:
        if key == "fitness_label":
            batch_out[key] = torch.stack([item[key] for item in batch])
        elif key.endswith("structure_coords"):
            max_len = max(item[key].shape[0] for item in batch)
            padded = []
            for item in batch:
                coord = item[key]  # (L, 3, 3)
                pad_len = max_len - coord.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len, 3, 3), float('nan'))
                    coord = torch.cat([coord, pad], dim=0)
                padded.append(coord)
            batch_out[key] = torch.stack(padded)  # (B, L, 3, 3)
        elif key.endswith("sequence_tokens"):
            max_len = max(item[key].shape[0] for item in batch)
            padded = []
            for item in batch:
                seq = item[key]  # (L,)
                pad_len = max_len - seq.shape[0]
                if pad_len > 0:
                    pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)  # pad id = 0
                    seq = torch.cat([seq, pad], dim=0)
                padded.append(seq)
            batch_out[key] = torch.stack(padded)  # (B, L)
        else:
            raise ValueError(f"Unknown batch key: {key}")

    return batch_out
