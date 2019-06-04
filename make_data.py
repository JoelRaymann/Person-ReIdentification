# For python 2 support
from __future__ import print_function, division, print_function, unicode_literals

# Import necessary packages
import argparse, sys, traceback

# Import in-house packages
import dataset_pair_gen

VERSION = dataset_pair_gen.__version__
DESCRIPTION = """
This is an API to generate the hard pair-wise dataset from the 
MARS Dataset for the Person ReID training.

Example Call would be
python make_data.py -I <input-dataset-path> -O <csv-save-path> -C <max count of dataset to generate> -S <checkpoint saves if needed>
"""

if __name__ == "__main__":

    # For help
    parser = argparse.ArgumentParser(description = DESCRIPTION)

    # Add options
    parser.add_argument("-V", "--version", help = "Shows program version", action = "store_true")
    parser.add_argument("-I", "--input-dir", help = "Input Dataset Directory to make the triplet data")
    parser.add_argument("-O", "--save-path", help = "The output csv path to store the data")
    parser.add_argument("-C", "--count", help = "The max count to create the dataset")
    parser.add_argument("-S", "--save-checkpoint", help = "Checkpoint saving if needed")

    # Read args
    args = parser.parse_args()

    # check for version
    if args.version:
        print("Using Version %s" %(VERSION))
        sys.exit(2)
    
    dataset_path = ""
    save_path = ""
    count = 0
    checkpoint = 0
    if not args.input_dir or not args.save_path or not args.count:
        print("{CRITICAL} Invalid arguments: please use help to know how to use")
        sys.exit(2)

    if args.input_dir:
        dataset_path = str(args.input_dir)
    if args.save_path:
        save_path = str(args.save_path)
    if args.count:
        count = int(args.count)
    if args.save_checkpoint:
        checkpoint = int(args.save_checkpoint)
    
    print("[INFO]: Creating dataset of count: %d from %s and saving in %s with checkpoints %d" %(count, dataset_path, save_path, checkpoint))
    dataset_pair_gen.dataset_pair_generator(dataset_path, output_csv_path = save_path, count = count, checkpoint = checkpoint)