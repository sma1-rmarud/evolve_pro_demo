import gradio as gr


def predict_gnn(f_type, structure_files):
    if not structure_files:
        return gr.Error("File upload failed.")

    file_names = [f.name for f in structure_files]
    for file_name in file_names:
        if f_type == 'CIF' and not file_name.endswith(".cif"):
            gr.Error("The file you uploaded is not in CIF format 💥!")
        elif f_type == 'PDB' and not file_name.endswith(".pdb"):
            gr.Error("The file you uploaded is not in PDB format 💥!")
