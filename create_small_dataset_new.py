import argparse
import os
import random
import shutil
import sys

import tqdm


def create_small_dataset(
    train_limit: int,
    valid_limit: int = 0,
    test_limit: int = 0,
    seed: int = -1,
):
    if seed != -1:
        random.seed(seed)

    INPUT_FOLDER = "data/training_input"
    TRUTH_FOLDER = "data/training_ground-truth"
    START_DIR = os.getcwd()

    # remove small_data if exists
    OUTPUT_DIR = os.path.join(START_DIR, "small_data")
    if os.path.isdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)

    CATEGORIES = ["training", "validation", "test"]

    # create output directories
    OUT_INPUT_DIRS = [
        os.path.join(OUTPUT_DIR, f"{p}_input") for p in CATEGORIES
    ]

    OUT_TRUTH_DIRS = [
        os.path.join(OUTPUT_DIR, f"{p}_truth") for p in CATEGORIES
    ]

    for input_dir, truth_dir in zip(OUT_INPUT_DIRS, OUT_TRUTH_DIRS):
        os.mkdir(input_dir)
        os.mkdir(truth_dir)

    LOWER_BOUNDS = [0, train_limit, train_limit + valid_limit]
    UPPER_BOUNDS = [
        train_limit,
        train_limit + valid_limit,
        train_limit + valid_limit + test_limit,
    ]

    # copy files
    files_to_cp = random.sample(
        os.listdir(INPUT_FOLDER), k=train_limit + valid_limit + test_limit
    )

    for i in range(3):
        lb = LOWER_BOUNDS[i]
        ub = UPPER_BOUNDS[i]
        in_dir = OUT_INPUT_DIRS[i]
        truth_dir = OUT_TRUTH_DIRS[i]

        for file in tqdm.tqdm(files_to_cp[lb:ub]):
            shutil.copyfile(
                os.path.join(INPUT_FOLDER, file),
                os.path.join(in_dir, file),
            )
            shutil.copyfile(
                os.path.join(TRUTH_FOLDER, file),
                os.path.join(truth_dir, file),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates new directories with a subset of original dataset.\n"
        'The output folders are always "./small_data/* '
        'IMPORTANT: make sure dataset is in "./data/training_input" and "./data/training_ground-truth" folders',
        epilog="Example usage: \n python3 create_small_dataset_new.py --train 4000 --valid 1000 --test 10000 -s 42",
    )
    parser.add_argument(
        "--train",
        help="Specifies how many training data to generate. Output is ./small_data/training_*",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--valid",
        help="Specifies how many validation data to generate. Output is ./small_data/validation_*",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--test",
        help="Specifies how many testing data to generate. Output is ./small_data/test_*",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed to use generating smaller dataset",
        default=-1,
    )
    args = parser.parse_args(sys.argv[1:])
    create_small_dataset(args.train, args.valid, args.test, args.seed)
