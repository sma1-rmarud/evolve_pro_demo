import os
import sys

from evolvepro.src.process import generate_single_aa_mutants, generate_wt

# sys.path.append(os.path.join(os.path.dirname(__file__), "../external/evolvepro"))


def predict_structure():
    pass


generate_wt(
    'ZKNVPQEPNZPCVZKNDINQORMVJNYUBNBIMZLZCXMVLQVPZXAUCYUBRMTVNZKDOKECOKPKQRNVV',
    output_file='kelsic_WT.fasta',
)
