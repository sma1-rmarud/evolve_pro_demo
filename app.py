import gradio as gr

def greet(f_type, structure_file):
    # request 날리는 줄 필요
    return f_type + structure_file

def upload_file(files):
    file_paths = [file.name for file in files]
    return file_paths

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center'>
        ES activity prediction using Deep Learning
        </h1>
        """,
        elem_id = "title"
    )
    
    file_format_choice = gr.Radio(
                ["PDB", "CIF"], 
                label="File format", 
                info="Choose your structure file format"
            )
    
    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)
    
    output_text = gr.Textbox(label="Predicted activity label")
    predict_button = gr.Button("predict")
    
    predict_button.click(
        fn=greet,
        inputs=[
            file_format_choice,
            file_output
        ],
        outputs=[output_text]
    )

demo.launch()