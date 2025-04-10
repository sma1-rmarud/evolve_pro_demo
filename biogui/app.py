import gradio as gr
import tempfile
import os

from biogui.utils.evolvepro_utils import predict_evolvepro
from biogui.utils.gnn_utils import predict_gnn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
custom_tmp = os.path.join(BASE_DIR, "tmp_gradio")
os.makedirs(custom_tmp, exist_ok=True)
tempfile.tempdir = custom_tmp
os.environ["GRADIO_TEMP_DIR"] = custom_tmp

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths


with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center'>
        🧬ES activity prediction using Deep Learning
        </h1>
        """,
        elem_id="title",
    )

    with gr.Tab(label='EvolvePro'):
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

    # with gr.Tab(label='Ours'):
    #     with gr.Row(variant='panel'):
    #         with gr.Column(scale=4):
    #             file_format_choice = gr.Radio(
    #                 ["PDB", "CIF"],
    #                 label="File format",
    #                 info="Choose your structure file format",
    #             )
    #         with gr.Column(scale=6):
    #             file_output = gr.Files(file_types=[".cif", ".pdb"])
    #             upload_button = gr.UploadButton(
    #                 "Click to Upload a File",
    #                 file_types=[".cif", ".pdb"],
    #                 file_count="multiple",
    #             )
    #             upload_button.upload(upload_file, upload_button, file_output)

    #     output_text = gr.Textbox(label="Predicted activity label")
    #     predict_button = gr.Button("predict")

    #     predict_button.click(
    #         fn=predict_gnn,
    #         inputs=[file_format_choice, file_output],
    #         outputs=[output_text],
    #     )

demo.launch()
