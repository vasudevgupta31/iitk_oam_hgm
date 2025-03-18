
 # **HGM - Hierarchical Generative Model**
 
 ## **Overview**
 HGM is a Hierarchical Generative Model designed for molecular generation and optimization using SMILES sequences. This project integrates deep learning techniques for data processing, training, beam search, and sampling, ensuring efficient molecular design.
 
 ## **Project Structure**
 ```
 hgm/
 │── configs/              # Configuration files for paths and parameters
 │── funcs/                # Helper functions for processing and training
 │── processes/            # Main scripts for data processing and model training
 │── results/              # Generated results and trained models
 │── README.md             # Project documentation
 │── requirements.txt      # Required dependencies
 │── main.py               # Entry point for running the pipeline
 ```
 
 ## **Installation**
 1. **Clone the repository**:
    ```bash
    git clone https://github.com/vasudevgupta31/iitk_oam_hgm.git
    cd hgm
    ```
 2. **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    venv\Scripts\activate     # Windows
    ```
 3. **Install dependencies**:
    ```bash
    pip install poetry
    poetry install
    ```
 
 ## **Usage**
 Modify `configs/config.ini` to set up:
 - input_file (File should be placed within `input_data`)
 - pretrained model (h5 file should be placed in /pretrained and the name should be given in config.ini file)
 - Trianing hyperparameters
 - other training options.
 
 ## **Logging & Debugging**
 The project utilizes `loguru` for logging. Logs are stored in `results/logs/` and provide insights into execution.
  
 ## **Contact**
 For queries or support, reach out at 
 `akshaykakkar.email@example.com`.
 `guptavasudelhi@gmail.com`
 `lokesh@domain.com`
