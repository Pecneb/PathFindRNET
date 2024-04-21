import sys
import argparse
from pathlib import Path
from typing import List
from joblib import dump
from itertools import starmap

from PathFindRNET.dataset.dataset_loader import load_dataset
from PathFindRNET.dataset.trajectory import TrackedObject

# Add the path to the trajectorynet module, this is the old repository path
sys.path.append(
    "/media/pecneb/970evoplus/gitclones/computer_vision_research/trajectorynet/"
)
from dataManagementClasses import TrackedObject as TO_Old


def get_args():
    parser = argparse.ArgumentParser(description="Migrate datasets from old to new.")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset file.",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to the output file.",
    )
    return parser.parse_args()


def from_old(old_obj: TO_Old) -> TrackedObject:
    """Convert an old tracked object to a new one.

    Parameters
    ----------
    old_obj : TO_Old
        Old tracked object.

    Returns
    -------
    TrackedObject
        New tracked object.
    """
    return TrackedObject.from_detections(old_obj.objID, old_obj.history, old_obj.max_age)


def convert_to_new(old_dataset: List[TO_Old]) -> List[TrackedObject]:
    """Convert an old dataset to a new one.

    Parameters
    ----------
    old_dataset : List[TO_Old]
        List of old tracked objects.

    Returns
    -------
    List[TrackedObject]
        List of new tracked objects.
    """
    ret = []
    for old_obj in old_dataset:
        print(old_obj)
        ret.append(from_old(old_obj))
        ret[-1].dataset = old_obj._dataset
    return ret


def main():
    args = get_args()
    src = Path(args.dataset_path)
    dst = Path(args.output_path)
    if dst.suffix != ".joblib" and not dst.exists():
        print("Destination is not a joblib file and directory does not exists, creating directory.")
        dst.mkdir(exist_ok=True, parents=True)
    if src.is_dir():
        for p in src.glob("*.joblib"):
            print(f"Converting {p.name}")
            old_dataset = load_dataset(p)
            new_dataset = convert_to_new(old_dataset)
            print(f"Saving to {dst / p.name}")
            dump(new_dataset, dst / p.name)
    else:
        old_dataset = load_dataset(src)
        new_dataset = convert_to_new(old_dataset)
        dump(new_dataset, dst)

if __name__ == "__main__":
    main()
