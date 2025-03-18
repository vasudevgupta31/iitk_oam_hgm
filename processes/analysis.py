import os
import time
import re
import joblib
import collections
from loguru import logger
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

from funcs import helper as hp
from configs import fixed_parameters as FP
from configs.path_config import (exp_name, 
                                 config_file, 
                                 exp_gen_samples_path,
                                 exp_output_path)


def novo_analysis():
    """
    Analyzes interim SMILES samples, validates molecules, and generates final filtered results.
    """
    start = time.time()
    
    # Get back the experiment parameters
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    min_len = int(config_file['PROCESSING']['min_len'])
    max_len = int(config_file['PROCESSING']['max_len'])
    temp = float(config_file['SAMPLING']['temp'])

    logger.info('\nSTART NOVO ANALYSIS')

    # Path to generated data
    path_gen = f'../results/{exp_name}/generated_data/'

    # Path to save analysis results
    save_path = os.path.join(exp_output_path, f'novo_molecules/Temp_{temp}/')
    os.makedirs(save_path, exist_ok=True)

    # Store abundance of all molecules
    d_abundance = collections.Counter()
    
    def process_file(filename):
        """Process a single file to extract valid SMILES."""
        name = filename.replace('.pkl', '')
        data = hp.load_obj(path_gen + filename)

        valids = []
        d_mol_count = collections.Counter()
        
        for gen_smile in data:
            if gen_smile and isinstance(gen_smile, str):
                # Optimize replacement using regex
                gen_smile = re.sub(f"[{pad_char}{end_char}{start_char}]", '', gen_smile)
                mol = Chem.MolFromSmiles(gen_smile)
                if mol:
                    cans = Chem.MolToSmiles(mol)
                    if len(cans) >= 1:
                        valids.append(cans)
                        d_mol_count[cans] += 1
                        d_abundance[cans] += 1
        
        return name, valids, d_mol_count

    t0 = time.time()
    
    # Parallel processing of all `.pkl` files
    results = joblib.Parallel(n_jobs=-1, backend="multiprocessing")(
        joblib.delayed(process_file)(filename)
        for filename in sorted(os.listdir(exp_gen_samples_path))
        if filename.endswith(f'{temp}.pkl')
    )

    for name, valids, d_mol_count in results:
        logger.info(f'Generated {len(valids)} SMILES in Epoch {name}')
        
        # Save sorted abundance data
        sorted_d_mol_count = sorted(d_mol_count.items(), key=lambda x: x[1], reverse=True)
        novo_name = os.path.join(save_path, f'molecules_{name}')
        
        with open(f'{novo_name}_abundance.txt', 'w') as f:
            f.writelines(f'{smi} \t {count}\n' for smi, count in sorted_d_mol_count)

        with open(f'{novo_name}.txt', 'w') as f:
            f.writelines(f"{item}\n" for item in valids)

    # Save total abundance
    sorted_d_abundance = sorted(d_abundance.items(), key=lambda x: x[1], reverse=True)
    novo_name = os.path.join(save_path, 'molecules')
    
    with open(f'{novo_name}_totalabundance_{temp}.txt', 'w') as f:
        f.writelines(f'{smi} \t {count}\n' for smi, count in sorted_d_abundance)

    logger.info(f'NOVO ANALYSIS COMPLETED in {time.time() - start:.2f} seconds')
