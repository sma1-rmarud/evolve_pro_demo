import re
import pandas as pd
from Bio import SeqIO
import sys
import subprocess
import os
import json
from pathlib import Path
from ours_model.inference import run as run_ours_model


def run_alphafold(protein_name: str,ligand_name: str):
    run_sh_path = "../../external/evolve_pro_demo_alphafold/alphafold3/run.sh"
    
    env = os.environ.copy()
    env["protein_name"] = protein_name
    env["ligand_name"] = ligand_name
    try:
        subprocess.run(
            ["bash", run_sh_path],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while running run.sh: {e}")

def run_post_alphafold(protein_name: str,ligand_name: str):
    run_sh_path = "./alphafold3_postprocess.sh"
    
    env = os.environ.copy()
    env["protein_name"] = protein_name
    env["ligand_name"] = ligand_name
    try:
        subprocess.run(
            ["bash", run_sh_path],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"Error while running run.sh: {e}")

def run_ours(protein_name, input_sequence_wt, ligand_name, ligand_smiles, csv_file):                        
    # RMMFAAAACIPLLLGSAPLYAQTSAVQQKLAALEKSSGGRLGVALIDTADNTQVLYRGDERFPMCSTSKVMAAAAVLKQSETQKQLLNQPVEIKPADLVNYNPIAEKHVNGTMTLAELSAAALQYSDNTAMNKLIAQLGGPGGVTAFARAIGDETFRLDRTEPTLNTAIPGDPRDTTTPRAMAQTLRQLTLGHALGETQRAQLVTWLKGNTTGAASIRAGLPTSWTVGDKTGSGDYGTTNDIAVIWPQGRAPLVLVTYFTQPQQNAESRRDVLASAARIIAEGL
    # CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C
    if len(csv_file) == 0:
        print("No files provided.")
        return ValueError("No files provided.")
    present_dir = os.path.dirname(os.path.abspath(__file__))
    print(present_dir)
    output_fasta = os.path.join(f"{present_dir}/result", f"{protein_name}.fasta")
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    sequences = []
    row_mutations = []

    wt_sequence = str(input_sequence_wt)
    header = (
            f">wt| "
    )
    sequences.append((header, "".join(wt_sequence)))    
    data = pd.read_csv(csv_file)
    
    for idx, row in enumerate(data['Variant']):
        print(row)
        mutations = []
        token = row.split(" ")
        print(token)
        for tk in token:
            m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", tk)
            print(m)
            if not m:
                raise ValueError(f"[row {idx}] 잘못된 변이 포맷: {tk}")
            orig, pos, mut = m.groups()
            pos = int(pos)
            mutations.append((orig, pos, mut))
        row_mutations.append(mutations)
    print(row_mutations)

    for i, muts in enumerate(row_mutations):
        seq = list(wt_sequence)
        data['wt_seq'] = wt_sequence
        L = len(seq)

        seen = {}
        for (orig, pos, mut) in muts:
            if pos < 1 or pos > L:
                raise IndexError(f"[row {i}] POS {pos} is out of range in length (1..{L})")
            if pos in seen and seen[pos] != (orig, mut):
                raise ValueError(f"[row {i}] Other mut is in POS {pos}: {seen[pos]} vs {(orig, mut)}")
            seen[pos] = (orig, mut)

        for (orig, pos, mut) in muts:
            current = seq[pos - 1]  # 1-based → 0-based
            if current != orig:
                raise ValueError(f"[row {i}] WT[{pos}]={current} != Expected original residue: {orig}")
            seq[pos - 1] = mut
            
        header = (
            f">fold_job_{i+1}| "
        )
        data['mut_seq'] = "".join(seq)

        sequences.append((header, "".join(seq)))


    with open(output_fasta, "w") as f:
        for header, seq in sequences:
            f.write(f"{header}\n{seq}\n")
    
    new_csv = f"{present_dir}/result", f"processed_{protein_name}_{ligand_name}.csv"
            
    data.to_csv(os.path.join(new_csv), index=False)

    json_output_dir = os.path.join(f"{present_dir}/result", f"{protein_name}_{ligand_name}_csv2json")
    os.makedirs(json_output_dir, exist_ok=True)

    structure_output_dir = os.path.join(f"{present_dir}/result", f"{protein_name}_{ligand_name}_structure_raw")
    os.makedirs(structure_output_dir, exist_ok=True)

    with open(output_fasta, "r") as f:
        lines = f.readlines()
        current_sequence = ""
        header = None
    
        for line in lines:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence and header:
                    header_name = header.split("|")[0][1:]
                    if ligand_name :
                        json_data = {
                            "name": header_name,
                            "modelSeeds": [42],
                            "sequences": [
                                {
                                    "protein": {
                                        "id": "A",
                                        "sequence": current_sequence,
                                        "unpairedMsa": None,
                                        "pairedMsa": None,
                                        "templates": []
                                    }
                                },
                                {
                                    "ligand": {
                                        "id": "B",
                                        "smiles": ligand_smiles
                                    }
                                }
                            ],
                            "dialect": "alphafold3",
                            "version": 1
                        }
                    else:
                        json_data = {
                            "name": header_name,
                            "modelSeeds": [42],
                            "sequences": [
                                {
                                    "protein": {
                                        "id": "A",
                                        "sequence": current_sequence,
                                        "unpairedMsa": None,
                                        "pairedMsa": None,
                                        "templates": []
                                    }
                                }
                            ],
                            "dialect": "alphafold3",
                            "version": 1
                        }

                    output_json = os.path.join(json_output_dir, f"{header_name}.json")
                    with open(output_json, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)

                header = line
                current_sequence = ""
            else:
                current_sequence += line

        if current_sequence and header:
            header_name = header.split("|")[0][1:]
            if ligand_name :
                json_data = {
                    "name": header_name,
                    "modelSeeds": [42],
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": current_sequence,
                                "unpairedMsa": None,
                                "pairedMsa": None,
                                "templates": []
                            }
                        },
                        {
                            "ligand": {
                                "id": "B",
                                "smiles": ligand_smiles
                            }
                        }
                    ],
                    "dialect": "alphafold3",
                    "version": 1
                }
            else:
                json_data = {
                    "name": header_name,
                    "modelSeeds": [42],
                    "sequences": [
                        {
                            "protein": {
                                "id": "A",
                                "sequence": current_sequence,
                                "unpairedMsa": None,
                                "pairedMsa": None,
                                "templates": []
                            }
                        }
                    ],
                    "dialect": "alphafold3",
                    "version": 1
                }


            output_json = os.path.join(json_output_dir, f"{header_name}.json")
            with open(output_json, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f"JSON 파일 생성 완료: {output_json}")
    if not ligand_name :
        ligand_name = "None"

    run_alphafold(protein_name, ligand_name)
    run_post_alphafold(protein_name, ligand_name)
    new_struct_dir = os.path.join(f"{present_dir}/result", f"{protein_name}_{ligand_name}_structure_processed")
    ckpt_path = "./chpt/esm3_finetuned.pth"
    out_csv = os.path.join(f"{present_dir}/result", f"{protein_name}_{ligand_name}_predicted.csv") # 다운로드 받을수 있게...
    run_ours_model(new_csv, new_struct_dir, ckpt_path, out_csv, device="cuda:0", bs=2, d_model=1536)