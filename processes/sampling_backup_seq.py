import os
import time
import warnings

import joblib
from loguru import logger

from funcs.helpers_training import load_model
from funcs.helpers_sampling import sample
from configs.path_config import (exp_name, 
                                 config_file, 
                                 exp_memory_path, 
                                 exp_gen_samples_path, 
                                 exp_models_path)
import configs.fixed_params as FP


def sampling(verbose=True):
    start = time.time()

    # Parameters to sample novo smiles
    temp = float(config_file['SAMPLING']['temp'])
    n_sample = int(config_file['SAMPLING']['n_sample'])
    if n_sample>5000:
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
    # start the sampling of new SMILES
    for epoch in range(start_epoch, end_epoch + 1):
        if not os.path.isfile(os.path.join(exp_models_path, f'{epoch:02d}_{temp}.pkl')):
            model_path = f'{exp_models_path}/{epoch:02d}.h5'
            model = load_model(model_path)
            logger.info(f'Sampling from model saved at epoch {epoch} with temp {temp}')
        
            generated_smi = []
            counter=0
            start_sampling = time.time()
            for n in range(n_sample):
                generated_smi.append(sample(model, temp, 
                                start_char, end_char, max_len+1, 
                                indices_token, token_indices))
        
            # From 100 molecules to sample,
            # we indicate the current status
            # to the user
                if n_sample>=100:
                    if len(generated_smi)%int(0.1*n_sample)==0:
                        counter+=10
                        delta_time = time.time()-start_sampling
                        start_sampling = start_sampling + delta_time
                        print(f'Status for model from epoch {epoch}: {counter}% of the molecules sampled in {delta_time:.2f} seconds')
        
            # hp.save_obj(generated_smi, f'{save_path}{epoch}_{temp}')
            joblib.dump(value=generated_smi, filename=os.path.join(exp_gen_samples_path, f'{epoch}_{temp}'))

        end = time.time()
        logger.info(f'SAMPLING DONE for model from epoch {epoch} in {end-start:.2f} seconds')  
