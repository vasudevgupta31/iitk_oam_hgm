"""
HGM - Hierarchical Generative Model Execution Pipeline

This script orchestrates the end-to-end execution of the HGM pipeline, including:
1. Data Processing: Prepares and processes input SMILES data.
2. Model Training: Trains a deep learning model on processed data.
3. Sampling: Generates molecular structures from the trained model.
4. Analysis: Evaluates and filters generated molecules.

Logging:
- Logs are saved in the `logs_path` directory with a timestamped filename.
- Log rotation is enabled at 10MB with retention for 10 days.

Usage:
Run this script as the main entry point to execute the pipeline:
```bash
python main.py
```
"""

import os
import datetime
from loguru import logger

from configs.path_config import (config_file, 
                                 exp_memory_path, 
                                 input_file_path, 
                                 clean_experiment_memory, 
                                 logs_path)
from processes.processing import preprocessing
from processes.training import train_network
from processes.sampling import generate_samples
from processes.analysis import novo_analysis


# Set up logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(logs_path, f"{timestamp}.log")
logger.add(log_file, rotation="10MB", level="INFO")


def main():

    # 0. Check Memory/interim path
    clean_experiment_memory()

    # 1. Data Processing
    logger.info("Starting data processing...")
    preprocessing(split=float(config_file['PROCESSING']['split']), 
                  input_data_file=input_file_path, 
                  augmentation=int(config_file['PROCESSING']['augmentation']), 
                  min_len=int(config_file['PROCESSING']['min_len']), 
                  max_len=int(config_file['PROCESSING']['max_len']), 
                  output_save_dir=exp_memory_path, 
                  verbose=True)
    logger.info("Data processing completed successfully.")

    # 2. Training
    logger.info("Starting network training...")
    train_network()
    logger.info("Network training completed successfully...")

    # 3. Sampling
    logger.info("Generating samples from trained network.")
    generate_samples()
    logger.info("Sample generation completed.")
    
    # 4. Analysis
    logger.info("Performing Novo analysis on generated samples.")
    novo_analysis()
    logger.info("Novo analysis completed succesfully.")


if __name__ == '__main__':
    main()
