import tensorflow as tf
import numpy as np
import time
import os
import warnings
import argparse
import configparser

import sys
import os


from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print("GPU detected. Configuring memory allocation...")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=38000)]
    )
    print("GPU memory allocation optimized!")
else:
    print("No GPU found! TensorFlow will use CPU.")

# Load fixed parameters and helper functions

# Assuming your 'python' module is in '/content/drive/MyDrive/GHIGC/Model2/multitarget-ligands/src/'
import sys
additional_paths = [
    '/usr/local/lib/python3.11/site-packages',
    '/content',
    '/env/python',
    '/usr/lib/python311.zip',
    '/usr/lib/python3.11',
    '/usr/lib/python3.11/lib-dynload',
    '/usr/local/lib/python3.11/dist-packages',
    '/usr/lib/python3/dist-packages',
    '/usr/local/lib/python3.11/dist-packages/IPython/extensions',
    '/root/.ipython'
]

for path in additional_paths:
    if path not in sys.path:
        sys.path.append(path)

from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='SMILES generation')
parser.add_argument('-c', '--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)

@tf.function
def one_hot_encode(token_lists, n_chars):
    return tf.one_hot(token_lists, n_chars)

@tf.function
def get_token_proba(preds, temp):
    preds = tf.cast(preds, tf.float32)
    preds = tf.math.log(preds) / temp
    probas = tf.nn.softmax(preds)
    return probas

@tf.function
def sample_step(model, seed_token, n_chars, temp):
    x_seed = one_hot_encode(tf.expand_dims(seed_token, 0), n_chars)
    preds = model(x_seed, training=False)
    logits = preds[0, -1, :]
    probas = get_token_proba(logits, temp)
    next_char_ind = tf.random.categorical(tf.math.log(probas[tf.newaxis, :]), num_samples=1)[0, 0]
    return next_char_ind

def sample(model, temp, start_char, end_char, max_len, indices_token, token_indices):
    n_chars = len(indices_token)
    seed_token = tf.constant([token_indices[start_char]], dtype=tf.int32)
    generated = [indices_token[str(seed_token.numpy()[0])]]
    
    while generated[-1] != end_char and len(generated) < max_len:
        next_char_ind = sample_step(model, seed_token, n_chars, temp)
        next_char = indices_token[str(next_char_ind.numpy())]
        generated.append(next_char)
        seed_token = tf.concat([seed_token, [next_char_ind]], axis=0)
    
    return ''.join(generated)

def batch_sample(model, temp, start_char, end_char, max_len, indices_token, token_indices, batch_size):
    generated_smi = []
    for _ in range(0, batch_size, 100):  # Process in sub-batches of 100
        batch_generated = [sample(model, temp, start_char, end_char, max_len, indices_token, token_indices) for _ in range(100)]
        generated_smi.extend(batch_generated)
    return generated_smi

if __name__ == '__main__':
    start = time.time()
    
    args = vars(parser.parse_args())
    verbose = args['verbose']
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    exp_name = configfile.split('/')[-1].replace('.ini', '')
    
    save_path = f'../results/{exp_name}/generated_data/'
    os.makedirs(save_path, exist_ok=True)
    dir_ckpts = f'../results/{exp_name}/models'
    
    temp = float(config['SAMPLING']['temp'])
    n_sample = int(config['SAMPLING']['n_sample'])
    start_epoch = int(config['SAMPLING']['start_epoch'])
    end_epoch = int(config['SAMPLING']['end_epoch'])
    
    max_len = int(config['PROCESSING']['max_len'])
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    indices_token = FP.INDICES_TOKEN
    token_indices = FP.TOKEN_INDICES
    
    if verbose: print('\nSTART SAMPLING')
    
    for epoch in range(start_epoch, end_epoch+1):
        if not os.path.isfile(f'{save_path}{epoch:02d}_{temp}.pkl'):
            model_path = f'{dir_ckpts}/{epoch:02d}.h5'
            model = tf.keras.models.load_model(model_path)
            
            if verbose: print(f'Sampling from model saved at epoch {epoch} with temp {temp}')
            
            batch_size = 1000  # Adjust based on your GPU memory
            generated_smi = []
            start_sampling = time.time()
            
            for i in range(0, n_sample, batch_size):
                batch_generated = batch_sample(model, temp, start_char, end_char, max_len+1, indices_token, token_indices, min(batch_size, n_sample-i))
                generated_smi.extend(batch_generated)
                
                if verbose and n_sample >= 100:
                    progress = min(100, (i + batch_size) / n_sample * 100)
                    delta_time = time.time() - start_sampling
                    print(f'Status for model from epoch {epoch}: {progress:.0f}% of the molecules sampled in {delta_time:.2f} seconds')
            
            hp.save_obj(generated_smi, f'{save_path}{epoch}_{temp}')
        
    end = time.time()
    if verbose: print(f'SAMPLING DONE for model from epoch {epoch} in {end-start:.2f} seconds')
