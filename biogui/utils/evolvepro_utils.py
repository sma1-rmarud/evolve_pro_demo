import os
import tempfile

from evolvepro.src.process import generate_single_aa_mutants, generate_wt


def predict_evolvepro(protein_name, input_sequence, num_runs):

    with tempfile.TemporaryDirectory() as temp_dir:
        wt_file = os.path.join(temp_dir, f"{protein_name}_WT.fasta")
        generate_wt(input_sequence, wt_file)

        mutant_file = os.path.join(temp_dir, f"{protein_name}.fasta")
        generate_single_aa_mutants(wt_file, mutant_file)

        mutant = open(mutant_file, "r").read().strip()

        for i in range(num_runs):
            mutant += f"\n#####\n{i}"

    return mutant
