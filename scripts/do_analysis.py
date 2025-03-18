import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='SMILES generation')
parser.add_argument('-c', '--configfile', type=str, help='Path to config file', required=True)
parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)


if __name__ == '__main__':
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    verbose = args['verbose']
    configfile = args['configfile']
    config = configparser.ConfigParser()
    config.read(configfile)
    exp_name = configfile.split('/')[-1].replace('.ini', '') 
    ####################################
    
    # get back the experiment parameters
    #mode = config['EXPERIMENTS']['mode']
    pad_char = FP.PROCESSING_FIXED['pad_char']
    start_char = FP.PROCESSING_FIXED['start_char']
    end_char = FP.PROCESSING_FIXED['end_char']
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    temp = float(config['SAMPLING']['temp'])
    # experiment parameters depending on the mode
    #augmentation = int(config['AUGMENTATION'][mode])
    
    if verbose: print('\nSTART NOVO ANALYSIS')
    
    ####################################        
    # Path to the generated data
    path_gen = f'../results/{exp_name}/generated_data/'
    
    # Path to save the novo analysis
    save_path = f'../results/{exp_name}/novo_molecules/Temp_{temp}/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    d_abundance = {}
    t0 = time.time()
    print(f"Near the for",path_gen)
    for filename in sorted(os.listdir(path_gen)):
        print("Inside for loop")
        print(filename)
    for filename in sorted(os.listdir(path_gen)):
        #print(f"Debug1 We are in directory", path_gen)
        if filename.endswith(f'{temp}.pkl'):
            name = filename.replace('.pkl', '')
            data = hp.load_obj(path_gen+filename)
            
            valids = []
            n_valid = 0
            d_mol_count = {}
            
            for gen_smile in data:
                
                if len(gen_smile)!=0 and isinstance(gen_smile, str):
                    gen_smile = gen_smile.replace(pad_char,'')
                    gen_smile = gen_smile.replace(end_char,'')
                    gen_smile = gen_smile.replace(start_char,'')
                    
                    mol = Chem.MolFromSmiles(gen_smile)
        #            print(f"Debug2", mol)
                    if mol is not None:
                        cans = Chem.MolToSmiles(mol)
                        if len(cans)>=1:
                            n_valid+=1
                            valids.append(cans)
                            if cans in d_mol_count:
                                d_mol_count[cans] += 1
                            else:
                                d_mol_count[cans] = 1
                            if cans in d_abundance:
                                d_abundance[cans] += 1
                            else:
                                d_abundance[cans] = 1
            # save abundance of the generated molecules
            sorted_d_mol_count = sorted(d_mol_count.items(), key=lambda x: x[1],reverse = True)
            print(f'Generated {n_valid} SMILES in Epoch {name}')
            # we save the novo molecules also as .txt
            novo_name = f'{save_path}molecules_{name}'
            with open(f'{novo_name}_abundance.txt', 'w+') as f:
                for smi, count in sorted_d_mol_count:
                    f.write(f'{smi} \t {count}\n')
                    #print(smi, count)
            with open(f'{novo_name}.txt', 'w+') as f:
                for item in valids:
                    f.write("%s\n" % item)
        else:
            print("No pkl file found")
    sorted_d_abundance = sorted(d_abundance.items(), key=lambda x: x[1],reverse = True)
    novo_name = f'{save_path}molecules'
    with open(f'{novo_name}_totalabundance_{temp}.txt', 'w+') as f:
        for smi, count in sorted_d_abundance:
            f.write(f'{smi} \t {count}\n')

           
