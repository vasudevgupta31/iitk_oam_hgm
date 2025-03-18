import os
import configparser
import shutil
from loguru import logger

root_path = os.getcwd()
ini_file_path = 'config.ini'
config_file = configparser.ConfigParser()
config_file.read("config.ini")

output_data_path = os.path.join(root_path, 'output_data')
os.makedirs(output_data_path, exist_ok=True)

logs_path = os.path.join(root_path, 'logs')
os.makedirs(logs_path, exist_ok=True)

# Pre-Trained Model
pretrained_model_path = os.path.join('pretrained', config_file['MODEL']['pretrained_model'])

# All child paths should be relative to the input file
input_file = config_file["INPUT"]["NameData"]
input_file_path = os.path.join("input_data", input_file)

# TO store multiruns and interim variables/states
memory_path = os.path.join(root_path, "memory")
os.makedirs(memory_path, exist_ok=True)

# Experiment name and Override
exp_name = config_file["INPUT"]["experiment_name"]
exp_memory_path = os.path.join(memory_path, exp_name)
os.makedirs(exp_memory_path, exist_ok=True)
overide_experiment = True if config_file["INPUT"]["override_experiment"].lower().startswith('y') else False

# Models path
exp_models_path = os.path.join(exp_memory_path, "models")
os.makedirs(exp_models_path, exist_ok=True)

# Gen-Samples path
exp_gen_samples_path = os.path.join(exp_memory_path, 'generated_samples')
os.makedirs(exp_gen_samples_path, exist_ok=True)

# Experiment results path
exp_output_path = os.path.join(output_data_path, exp_name)
os.makedirs(exp_output_path, exist_ok=True)


def clean_experiment_memory():
    if exp_name in os.listdir(memory_path):
        if not overide_experiment:
            msg = f"Experiment named: `{exp_name}` already exists and override is set to `N`. In the confi.ini file, please change the experiment name to make a new experiment or change override to `Y` to redo with the same name."
            logger.error(msg)
            raise ValueError(msg)
        else:
            shutil.rmtree(exp_memory_path)
            shutil.rmtree(exp_models_path)
            shutil.rmtree(exp_gen_samples_path)
            shutil.rmtree(exp_output_path)
            
            os.makedirs(exp_memory_path)
            os.makedirs(exp_models_path)
            os.makedirs(exp_gen_samples_path)
            os.makedirs(exp_output_path)
