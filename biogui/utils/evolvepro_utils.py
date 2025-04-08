import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), "../external/evolvepro"))

from evolvepro.src.process import generate_wt, generate_single_aa_mutants

def predict_structure():
    pass

generate_wt('ZKNVPQEPNZPCVZKNDINQORMVJNYUBNBIMZLZCXMVLQVPZXAUCYUBRMTVNZKDOKECOKPKQRNVV', output_file='kelsic_WT.fasta')