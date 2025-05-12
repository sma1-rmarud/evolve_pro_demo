import os
import tempfile
from functools import partial

import gradio as gr

from biogui.utils.evolvepro_utils import predict_evolvepro, exp_process

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

        with gr.Tab(label='EvolvePro_run_n_mutants'):
            gr.Markdown("## 🔬 Generate N-mutant Combinations")
            
            wt_seq = gr.Textbox(label="Wildtype Sequence", placeholder="MMA...")
            mutant_file = gr.File(file_types=[".xlsx"], label="Upload Mutant Excel File")
            n_mutant = gr.Dropdown([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], label="Number of Mutations (n)", value=1)
            threshold = gr.Number(label="Activity Threshold", value=1.0)
            
            output_file = gr.File(label="Generated mutant FASTA")

            generate_button = gr.Button("Generate Mutants")
            exp_process = partial(
                exp_process,
                gradio_tmp=gradio_tmp,
            )
            generate_button.click(
                fn=exp_process,
                inputs=[wt_seq, mutant_file, n_mutant, threshold],
                outputs=[output_file]
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
                value=512,
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
                    round_files,
                ],
                outputs=[output_img],
            )

    demo.launch(debug=True)