import os
import tempfile
from pathlib import Path
from argparse import Namespace
from itertools import combinations

import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from evolvepro.src.evolve import evolve_experimental
from evolvepro.plm.esm.extract import concatenate_files
from evolvepro.plm.esm.extract import run as extract_embeddings
from evolvepro.src.plot import plot_variants_by_iteration, read_exp_data
from evolvepro.src.process import generate_wt, generate_single_aa_mutants, generate_n_mutant_combinations

def exp_process(wt_seq, mutant_file, n_mutant, threshold, gradio_tmp):
    project_root = gradio_tmp
    wt_fasta = os.path.join(project_root, 'data', 'exp', 'wt_fasta', 'WT.fasta')
    os.makedirs(os.path.dirname(wt_fasta), exist_ok=True)
    generate_wt(wt_seq, wt_fasta)

    output_file = os.path.join(project_root, 'output', 'exp', 'mutant.fasta')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_single_aa_mutants(wt_fasta, output_file)
    
    generate_n_mutant_combinations(wt_fasta, mutant_file, n=n_mutant, output_file=output_file, threshold=threshold)

    return output_file

def predict_evolvepro(
    protein_name,
    input_sequence,
    num_rounds,
    embedding_model,
    toks_per_batch,
    number_of_variants,
    round_files,
):

    if len(round_files) != int(num_rounds):
        raise ValueError(
            "Number of round files must match the number of rounds."
        )

    round_files = sorted([Path(file.name) for file in round_files])

    with tempfile.TemporaryDirectory() as temp_dir:

        logger.info(f"Creating Wildtype sequence file for {protein_name}")
        wt_file = os.path.join(temp_dir, f"{protein_name}_WT.fasta")
        generate_wt(input_sequence, wt_file)

        logger.info(f"Creating single AA mutants for {protein_name}")
        fasta_file = os.path.join(temp_dir, f"{protein_name}.fasta")
        generate_single_aa_mutants(wt_file, fasta_file)

        output_dir = os.path.join(temp_dir, "output", embedding_model)
        concatenate_dir = os.path.join(temp_dir, "output")
        args = Namespace(
            model_location=embedding_model,
            fasta_file=fasta_file,
            output_dir=Path(output_dir),
            toks_per_batch=toks_per_batch,
            repr_layers=[-1],
            include=["mean"],
            truncation_seq_length=1022,
            nogpu=False,
            concatenate_dir=concatenate_dir,
        )
        logger.info(f"Extracting embeddings with args: {args}")
        extract_embeddings(args)

        fasta_file_name = Path(args.fasta_file).stem
        output_csv = f"{args.concatenate_dir}/{fasta_file_name}_{args.model_location}.csv"
        concatenate_files(args.output_dir, output_csv)

        embeddings_base_path = concatenate_dir
        embeddings_file_name = f"{fasta_file_name}_{args.model_location}.csv"
        round_base_path = os.path.join(temp_dir, "rounds_data")
        number_of_variants = int(number_of_variants)
        rename_WT = False
        output_dir = embeddings_base_path
        wt_fasta_path = Path(wt_file)

        round_file_names = round_files
        round_name = f'Round{num_rounds}'

        this_round_variants, df_test, df_sorted_all = evolve_experimental(
            protein_name,
            round_name,
            embeddings_base_path,
            embeddings_file_name,
            round_base_path,
            round_file_names,
            wt_fasta_path,
            rename_WT,
            number_of_variants,
            output_dir,
        )

        img_output_dir = os.path.join(temp_dir, "output", "images")
        img_output_file = f"{protein_name}_{embedding_model}.png"
        df = read_exp_data(round_base_path, round_file_names, wt_fasta_path)
        plot_variants_by_iteration(
            df,
            activity_column='activity',
            output_dir=img_output_dir,
            output_file=img_output_file,
        )

        img_path = os.path.join(
            img_output_dir, f"{img_output_file}_by_iteration.png"
        )
        img = Image.open(img_path)

    return img