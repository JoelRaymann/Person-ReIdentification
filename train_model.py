# For Python 2 support
from __future__ import absolute_import, division, print_function, unicode_literals

# import necessary functions
import sys, os, inspect, argparse, yaml

# Import in-house functions
import os_utils.os_utilities as os_utilities
import train_utils.train_utilities as train_utilities


VERSION = "0.1 alpha"
DESCRIPTION = """
This is an API to train the proposed Person ReID model on the given dataset of images from MARS 
image dataset and save it in the mentioned saving directory. The API offers fresh and resumed
training for the model. Please,check for model_utils folder and train_utils folder before
running this script to get error free execution

Example Call would be
Fresh Train:
    python train_model.py -C <config yaml file path>

Resume Train:
    python train_model.py -L <load model file path> -R <resume epoch> -C <config yaml file path>
"""

if __name__ == "__main__":

    # For help 
    parser = argparse.ArgumentParser(description = DESCRIPTION)
    
    # Add options
    parser.add_argument("-V", "--version", help = "Shows program version", action = "store_true")
    parser.add_argument("-L", "--load-model", help = "The model h5 file to load")
    parser.add_argument("-R", "--resume-epoch", help = "The epoch to resume train")
    parser.add_argument("-C", "--config-file", help = "The YAML config file for training")
    # Read args
    args = parser.parse_args()

    # check for version
    if args.version:
        print("Using Version %s" %(VERSION))
        sys.exit(2)
    
    load_model_path = ""
    resume_epoch = 0
    config_file_path = "config.yaml"
    
    if args.load_model:
        load_model_path = str(args.load_model)
    if args.resume_epoch:
        resume_epoch = int(args.resume_epoch)
    if args.config_file:
        config_file_path = str(args.config_file)
    
    with open(config_file_path, "r") as file:
        config = yaml.load(stream = file)


    if load_model_path == "":
        train_utilities.new_train_model(config = config)
    
    else:
        train_utilities.deterred_train_model(model_path = load_model_path, resume_epoch = resume_epoch, config = config)