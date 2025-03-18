import os
import time
import warnings
import joblib
from loguru import logger
from joblib import Parallel, delayed

from funcs.helpers_training import load_model
from funcs.helpers_sampling import sample
from configs.path_config import (exp_name, 
                                 config_file, 
                                 exp_memory_path, 
                                 exp_gen_samples_path, 
                                 exp_models_path)
import configs.fixed_params as FP


def generate_samples():
    """
    Performs SMILES sequence sampling using a trained model.

    This function loads pre-trained models from different epochs and generates 
    SMILES sequences using temperature-based sampling. The generated sequences 
    are saved to a specified directory.
    
    Returns:
        None: Saves the generated SMILES sequences to files.
    """
    start = time.time()

    # Parameters for SMILES sampling
    temp = float(config_file['SAMPLING']['temp'])
    n_sample = int(config_file['SAMPLING']['n_sample'])
    if n_sample > 5000:
        warnings.warn('You will sample more than 5000 SMILES; this will take a while')

    start_epoch = int(config_file['SAMPLING']['start_epoch'])
    end_epoch = int(config_file['SAMPLING']['end_epoch'])

    max_len = int(config_file['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES

    logger.info('\nSTART SAMPLING')

    # Start the sampling of new SMILES
    for epoch in range(start_epoch, end_epoch + 1):
        save_path = os.path.join(exp_gen_samples_path, f'{epoch}_{temp}')
        if not os.path.isfile(save_path):
            model_path = f'{exp_models_path}/{epoch:02d}.h5'
            model = load_model(model_path)
            logger.info(f'Sampling from model at epoch {epoch} with temp {temp}')

            start_sampling = time.time()
            
            # Use joblib for parallel sampling
            generated_smi = Parallel(n_jobs=-1, backend="multiprocessing")(
                delayed(sample)(model, temp, start_char, end_char, max_len + 1, indices_token, token_indices)
                for _ in range(n_sample)
            )

            # Save results
            joblib.dump(value=generated_smi, filename=save_path)

            delta_time = time.time() - start_sampling
            logger.info(f'SAMPLING DONE for model from epoch {epoch} in {delta_time:.2f} seconds')

    end = time.time()
    logger.info(f'Total Sampling Time: {end - start:.2f} seconds')
