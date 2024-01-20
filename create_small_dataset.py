import argparse
import os
import random
import shutil
import sys

import tqdm


def create_small_dataset(
    input_folder: str,
    ground_truth_folder: str,
    limit: int,
    seed: int,
):
    if seed != -1:
        random.seed(seed)

    START_DIR = os.getcwd()

    # create output directories
    OUT_INPUT_DIR = START_DIR + "/data/min_input/"
    OUT_TRUTH_DIR = START_DIR + "/data/min_ground-truth/"
    if os.path.isdir(OUT_INPUT_DIR):
        shutil.rmtree(OUT_INPUT_DIR)
    if os.path.isdir(OUT_TRUTH_DIR):
        shutil.rmtree(OUT_TRUTH_DIR)
    os.mkdir(OUT_INPUT_DIR)
    os.mkdir(OUT_TRUTH_DIR)

    # copy files
    files_to_cp = random.sample(os.listdir(input_folder), k=limit)
    for file in files_to_cp:
        shutil.copyfile(
            os.path.join(input_folder, file),
            os.path.join(OUT_INPUT_DIR, file),
        )
        shutil.copyfile(
            os.path.join(ground_truth_folder, file),
            os.path.join(OUT_TRUTH_DIR, file),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates new directories with a subset of original dataset.\n"
        'The output folders are always "./data/min_input/" and "./data/max_input"',
        epilog="Example usage: \n python3 create_small_dataset.py -i data/training_input -t data/training_ground-truth -n 100 -s 42",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input data",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--truth",
        help="Path to ground-truth data",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        help="Number of images to select",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed to use generating smaller dataset",
        default=-1,
    )
    args = parser.parse_args(sys.argv[1:])
    create_small_dataset(args.input, args.truth, args.limit, args.seed)
