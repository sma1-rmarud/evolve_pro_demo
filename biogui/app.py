import gradio as gr

from biogui.utils.evolvepro_utils import predict_evolvepro


def predict(f_type, structure_files):
    if not structure_files:
        return gr.Error("File upload failed.")

    file_names = [f.name for f in structure_files]
    for file_name in file_names:
        if f_type == 'CIF' and not file_name.endswith(".cif"):
            gr.Error("The file you uploaded is not in CIF format 💥!")
        elif f_type == 'PDB' and not file_name.endswith(".pdb"):
            gr.Error("The file you uploaded is not in PDB format 💥!")


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
            fn=predict_evolvepro,
            inputs=[file_format_choice, file_output],
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
            fn=predict,
            inputs=[file_format_choice, file_output],
            outputs=[output_text],
        )

demo.launch()
