# gradio_utils.py

import math
import pandas as pd
import gradio as gr

def update_n_and_length(wt_seq):
    length = len(wt_seq.strip())
    if length == 0:
        return gr.update(choices=[], value=None), ""
    max_n = max(length, 10)
    return gr.update(choices=list(range(1, max_n + 1)), value=1), str(length)
