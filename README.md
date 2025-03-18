 # **HGM - Hierarchical Generative Model**
 
 ## **Overview**
 HGM is a Hierarchical Generative Model designed for molecular generation and optimization using SMILES sequences. This project integrates deep learning techniques for data processing, training, beam search, and sampling, ensuring efficient molecular design.
 
 ## **Project Structure**
 ```
 hgm/
 │── configs/              # Configuration files for model and pipeline settings
 │── input_data/           # Raw input files for training
 │── output_data/          # Generated resultant molecules per experiment
 │── pretrained/           # Pretrained model files (.h5) for transfer learning
 │── funcs/                # Utility functions for data processing and model handling
 │── processes/            # processes abstraction for different routines in 
 │── memory/               # Generated interim files, models etc for the experiment
 │── README.md             # Project documentation
 │── pyproject.toml        # Poetry configuration file for dependency management
 │── main.py               # Entry point for executing the full pipeline
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
 - pretrained model (h5 file should be placed in /pretrained and the name should be given in config.ini file) (you can download from - https://drive.google.com/file/d/1hyMgwQnU9V7u5cKER9dSS_0pwJneWluj/view?usp=drive_link)
 - Trianing hyperparameters
 - other training options.

 ```
 # Add an input file (my_input.txt) to /input_data
 # In confi.ini - [INPUT] - NameData change value to `my_input.txt`
 # In confi.ini - [INPUT] - experiment_name change to `my_experiment`
 # In terminal run > python main.py
 # You will see results in /output_data under the folder `my_experiment`
 ```
 
 ## **Logging & Debugging**
 The project utilizes `loguru` for logging. Logs are stored in `results/logs/` and provide insights into execution.
  
 ## **Contact**
 For queries or support, reach out at 
 `akshaykakkar.email@example.com`.
 `guptavasudelhi@gmail.com`
 `lokesh@domain.com`
