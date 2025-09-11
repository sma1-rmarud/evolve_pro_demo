import re
import pandas as pd
from Bio import SeqIO
import sys
import subprocess
import os
import json
from pathlib import Path

# def run_alphafold(protein_name: str):
#     present_dir = os.path.dirname(os.path.abspath(__file__))
#     run_sh_path = "/evolve_pro_demo/external/evolve_pro_demo_alphafold/alphafold3/run.sh"
    
#     env = os.environ.copy()
#     env["protein_name"] = protein_name
    
#     try:
#         subprocess.run(
#             ["bash", run_sh_path],
#             check=True,
#             env=env
#         )
#     except subprocess.CalledProcessError as e:
#         print(f"Error while running run.sh: {e}")


def run_ours(protein_name, input_sequence_wt, ligand_smiles, csv_file):                        
    # RMMFAAAACIPLLLGSAPLYAQTSAVQQKLAALEKSSGGRLGVALIDTADNTQVLYRGDERFPMCSTSKVMAAAAVLKQSETQKQLLNQPVEIKPADLVNYNPIAEKHVNGTMTLAELSAAALQYSDNTAMNKLIAQLGGPGGVTAFARAIGDETFRLDRTEPTLNTAIPGDPRDTTTPRAMAQTLRQLTLGHALGETQRAQLVTWLKGNTTGAASIRAGLPTSWTVGDKTGSGDYGTTNDIAVIWPQGRAPLVLVTYFTQPQQNAESRRDVLASAARIIAEGL
    # CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)[C@@H](C3=CC=CC=C3)N)C(=O)O)C
    if len(csv_file) == 0:
        print("No files provided.")
        return ValueError("No files provided.")
    present_dir = os.path.dirname(os.path.abspath(__file__))
    print(present_dir)
    output_fasta = os.path.join(f"{present_dir}/result", f"{protein_name}.fasta")
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    
    wt_sequence = str(input_sequence_wt)
    data = pd.read_csv(csv_file)
    mutated_sequences = []
    row_mutations = []
    
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
        

        mutated_sequences.append((header, "".join(seq)))


    with open(output_fasta, "w") as f:
        for header, seq in mutated_sequences:
            f.write(f"{header}\n{seq}\n")

    json_output_dir = os.path.join(f"{present_dir}/result", f"{protein_name}_csv2json")
    os.makedirs(json_output_dir, exist_ok=True)

    structure_output_dir = os.path.join(f"{present_dir}/result", f"{protein_name}_structure")
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
                    combined_name = f"{os.path.splitext(os.path.basename(output_fasta))[0]}_{header_name}"

                    json_data = {
                        "name": combined_name,
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

                    output_json = os.path.join(json_output_dir, f"{combined_name}.json")
                    with open(output_json, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)

                header = line
                current_sequence = ""
            else:
                current_sequence += line

        if current_sequence and header:
            header_name = header.split("|")[0][1:]
            combined_name = f"{os.path.splitext(os.path.basename(output_fasta))[0]}_{header_name}"

            json_data = {
                "name": combined_name,
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

            output_json = os.path.join(json_output_dir, f"{combined_name}.json")
            with open(output_json, "w") as json_file:
                json.dump(json_data, json_file, indent=4)

            print(f"JSON 파일 생성 완료: {output_json}")
            
    
            
        
            
            

