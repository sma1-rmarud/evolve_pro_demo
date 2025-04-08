# This code is based on the following Colab notebook:
# https://colab.research.google.com/drive/1YCWvR73ItSsJn3P89yk_GY1g5GEJUlgy?usp=sharing

import os
import tempfile
from argparse import Namespace
from pathlib import Path

from evolvepro.plm.esm.extract import concatenate_files
from evolvepro.plm.esm.extract import run as extract_embeddings
from evolvepro.src.evolve import evolve_experimental
from evolvepro.src.plot import plot_variants_by_iteration, read_exp_data
from evolvepro.src.process import generate_single_aa_mutants, generate_wt
from loguru import logger
from PIL import Image


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
