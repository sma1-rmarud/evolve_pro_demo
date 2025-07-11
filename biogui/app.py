import os
import tempfile
from functools import partial
import pandas as pd

import gradio as gr

from biogui.utils.evolvepro_utils import predict_evolvepro, predict_n_mutants

with tempfile.TemporaryDirectory() as gradio_tmp:
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            <h1 style='text-align: center'>
            🧬ES activity prediction using Deep Learning
            </h1>
            """,
            elem_id="title",
        )

        with gr.Tab(label='EvolvePro_run_round'):
            gr.Markdown("## 🔬 Run EvolvePro Evolution Round ")
            protein_name = gr.Textbox(
                label="protein name", placeholder="Enter protein name"
            )
            input_sequence = gr.Textbox(
                label="sequence to generate wildtype",
                placeholder="Enter sequence to generate wildtype",
            )
            embedding_model = gr.Radio(
                ["esm1b_t33_650M_UR50S", "esm2_t36_3B_UR50D"],
                label="embedding model",
                info="Choose your embedding model",
                value="esm1b_t33_650M_UR50S",
            )
            toks_per_batch = gr.Slider(
                label="tokens per batch",
                value=4096,
                minimum=1,
                maximum=4096,
                step=1,
                info="Choose the number of tokens per batch for embedding extraction",
            )
            num_rounds = gr.Dropdown(
                list(range(1, 20)),
                label="number of rounds",
                info="Choose the number of rounds for evolution",
                value="1",
            )
            number_of_variants = gr.Dropdown(
                list(range(1, 20)),
                label="number of variants",
                info="Choose the number of variants for evolution",
                value="12",
            )
            round_files = gr.Files(
                file_types=[".xlsx"],
                label="experimental round activity files",
                file_count="multiple",
            )

            output_img = gr.Image(label="output image")
            
            output_table_variants = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Tested variants in this round")
            
            output_table_test = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Top variants predicted by the model")
            
            output_table_sorted = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Sorted all(train+test) the variants with y_pred")

            predict_button = gr.Button("predict")

            predict_button.click(
                fn=predict_evolvepro,
                inputs=[
                    protein_name,
                    input_sequence,
                    num_rounds,
                    embedding_model,
                    toks_per_batch,
                    number_of_variants,
                    round_files
                ],
                outputs=[output_img, output_table_variants, output_table_test, output_table_sorted],
            )
            
        with gr.Tab(label='EvolvePro_run_n_mutants'):
            gr.Markdown("## 🔬 Generate and Predict N-Mutant Combinations")

            wt_seq = gr.Textbox(label="Wildtype Sequence", placeholder="MMA...")

            mutant_file = gr.Files(
                file_types=[".xlsx"],
                label="Mutant activity Excel (with Variant, Activity)",
                file_count="multiple",
            )

            n_mutant = gr.Dropdown([1, 2, 3, 4, 5], label="Number of Mutations (n)", value=3)
            threshold = gr.Number(label="Activity Threshold", value=0.7)
            embedding_model = gr.Radio(
                ["esm1b_t33_650M_UR50S", "esm2_t36_3B_UR50D"],
                label="Embedding Model",
                value="esm1b_t33_650M_UR50S"
            )
            toks_per_batch = gr.Slider(minimum=1, maximum=4096, value=4096, label="Tokens per batch")
            number_of_variants = gr.Slider(minimum=1, maximum=50, value=12, label="Top-N to select")

            output_img = gr.Image(label="output image")
            
            output_table_variants = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Tested variants in this round")
            
            output_table_test = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Top variants predicted by the model")
            
            output_table_sorted = gr.Dataframe(
                                        value=pd.DataFrame(),
                                        headers=[],
                                        datatype=[],
                                        col_count=(0, "fixed"),
                                        row_count=(0, "fixed"),
                                        label="Sorted all(train+test) the variants with y_pred")

            predict_button = gr.Button("generate ansd predict")
            predict_button.click(
                fn=predict_n_mutants,
                inputs=[
                    wt_seq, mutant_file, n_mutant, threshold,
                    embedding_model, toks_per_batch, number_of_variants
                ],
                outputs=[
                    output_img, output_table_variants, output_table_test, output_table_sorted
                ]
            )




    demo.launch(debug=True)