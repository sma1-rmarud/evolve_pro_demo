import gradio as gr

from biogui.utils.evolvepro_utils import predict_evolvepro
from biogui.utils.gnn_utils import predict_gnn


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
        num_runs = gr.Slider(
            label="number of runs", value=1, minimum=1, maximum=10, step=1
        )

        output_text = gr.Textbox(label="Predicted activity label")
        predict_button = gr.Button("predict")

        predict_button.click(
            fn=predict_evolvepro,
            inputs=[protein_name, input_sequence, num_runs],
            outputs=[output_text],
        )

    with gr.Tab(label='Ours'):
        with gr.Row(variant='panel'):
            with gr.Column(scale=4):
                file_format_choice = gr.Radio(
                    ["PDB", "CIF"],
                    label="File format",
                    info="Choose your structure file format",
                )
            with gr.Column(scale=6):
                file_output = gr.Files(file_types=[".cif", ".pdb"])
                upload_button = gr.UploadButton(
                    "Click to Upload a File",
                    file_types=[".cif", ".pdb"],
                    file_count="multiple",
                )
                upload_button.upload(upload_file, upload_button, file_output)

        output_text = gr.Textbox(label="Predicted activity label")
        predict_button = gr.Button("predict")

        predict_button.click(
            fn=predict_gnn,
            inputs=[file_format_choice, file_output],
            outputs=[output_text],
        )

demo.launch()
