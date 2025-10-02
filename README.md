# ES activity prediction using Deep Learning

We made gradio demo for evolvepro, we hope this can be used for progress in bio research

## Installation

```bash
conda create -n bio python=3.10 -y
conda activate bio

git clone --recurse-submodules https://github.com/sma1-rmarud/evolve_pro_demo.git
cd evolve_pro_demo
git submodule update --remote --merge

pip install -e .

cd biogui/utils
python get_ckpts.py

cd external/evolve_pro_demo_alphafold
mkdir Database
cd alphafold3
chmod u+x fetch_databases.sh
./fetch_databases.sh ../Database

```

## Usage
[warning] You need Docker to run our finetuned esm3 model
To run the GUI server, execute: 
```bash
./run_gui.sh
```

<img width="946" alt="스크린샷 2025-04-08 오후 7 33 57" src="https://github.com/user-attachments/assets/fc69f343-ea4b-4a14-aa87-ae6f42277d76" />
