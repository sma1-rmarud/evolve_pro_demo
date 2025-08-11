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

from evolvepro.src.evolve import evolve_experimental, evolve_experimental_multi
from evolvepro.plm.esm.extract import concatenate_files
from evolvepro.plm.esm.extract import run as extract_embeddings
from evolvepro.src.plot import plot_variants_by_iteration, read_exp_data
from evolvepro.src.process import generate_wt, generate_single_aa_mutants, generate_n_mutant_combinations

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
        this_round_variants_df = this_round_variants.to_frame(name="variant")

    return  img, this_round_variants_df, df_test, df_sorted_all

    
def predict_n_mutants(
    protein_name,
    input_sequence,
    s_mutant_file,
    m_mutant_file,
    n_mutant,
    threshold,
    embedding_model,
    toks_per_batch,
    number_of_variants,
):
    round_files = sorted([Path(file) for file in s_mutant_file])
    if not m_mutant_file:
        m_round_files = []
    else:
        m_round_files = sorted([Path(file) for file in m_mutant_file])

    with tempfile.TemporaryDirectory() as temp_dir:
        embeddings_file_names = []
        logger.info(f"Creating Wildtype sequence file for {protein_name}")
        wt_file = os.path.join(temp_dir, f"{protein_name}_WT.fasta")
        generate_wt(input_sequence, wt_file)

        logger.info(f"Creating single AA mutants for {protein_name}")
        fasta_file = os.path.join(temp_dir, f"{protein_name}.fasta")
        generate_single_aa_mutants(wt_file, fasta_file)

        round_data_dir = os.path.join(temp_dir, "rounds_data")
        os.makedirs(round_data_dir, exist_ok=True)
        
        embedding_dir = os.path.join(temp_dir, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        
        merged_df_list = []
        merged_output_path = os.path.join(round_data_dir, "merged_mutants.xlsx")
        
        for f in round_files:
            df = pd.read_excel(f)
            df = df[["Variant", "activity"]]
            
            merged_df_list.append(df)

        if merged_df_list:
            merged_df = pd.concat(merged_df_list, ignore_index=True)
            merged_df.to_excel(merged_output_path, index=False)
            
        concatenate_dir = os.path.join(temp_dir, "output")
        args_single = Namespace(
            model_location=embedding_model,
            fasta_file=fasta_file,
            output_dir=Path(os.path.join(embedding_dir, "single_mutants_emb")),
            toks_per_batch=toks_per_batch,
            repr_layers=[-1],
            include=["mean"],
            truncation_seq_length=1022,
            nogpu=False,
            concatenate_dir=concatenate_dir,
        )
        extract_embeddings(args_single)
        
        single_fasta_name = f"single_{protein_name}_fasta"
        single_embedding_csv_name = f"{single_fasta_name}_{embedding_model}.csv"
        concatenate_files(args_single.output_dir, os.path.join(embedding_dir, single_embedding_csv_name))
        embeddings_file_names.append(single_embedding_csv_name)
        for n in range(n_mutant, 1, -1):
            n_mutant_fasta = os.path.join(temp_dir, f"dataset_{n}th.fasta")
            generate_n_mutant_combinations(
                wt_file, merged_output_path, n=n,
                output_file=n_mutant_fasta, threshold=threshold,
            )
            
            args_n_mutant = Namespace(
                model_location=embedding_model,
                fasta_file=n_mutant_fasta,
                output_dir=Path(os.path.join(embedding_dir, f"n_mutants_emb_{n}")),
                toks_per_batch=toks_per_batch,
                repr_layers=[-1],
                include=["mean"],
                truncation_seq_length=1022,
                nogpu=False,
                concatenate_dir=concatenate_dir,
            )
            extract_embeddings(args_n_mutant)

            n_mutant_fasta_name = f"n_mutant_{protein_name}_fasta{n}"
            n_mutant_embedding_csv_name = f"{n_mutant_fasta_name}_{embedding_model}.csv"
            concatenate_files(args_n_mutant.output_dir, os.path.join(embedding_dir, n_mutant_embedding_csv_name))
            embeddings_file_names.append(n_mutant_embedding_csv_name)
        
        this_round_variants, df_test, df_sorted_all = evolve_experimental_multi(
            protein_name=protein_name,
            round_name=f"Round_n{n_mutant}",
            embeddings_base_path=embedding_dir,
            embeddings_file_names=embeddings_file_names,
            round_base_path=round_data_dir,
            round_file_names_single=round_files,
            round_file_names_multi=m_round_files,
            wt_fasta_path=wt_file,
            rename_WT=False,
            number_of_variants=int(number_of_variants),
            output_dir=embedding_dir,
        )
        img_output_dir = os.path.join(temp_dir, "output", "images")
        img_output_file = f"N_Mutant_{protein_name}_{embedding_model}.png"
        df = read_exp_data(round_data_dir, round_files, wt_file)
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
        this_round_variants_df = this_round_variants.to_frame(name="variant")

        return  img, this_round_variants_df, df_test, df_sorted_all
