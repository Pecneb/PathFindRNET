### System Imports ###
from typing import Union, List
from pathlib import Path

### Third-Party Imports ###
import numpy as np
import tqdm
import joblib


def loadDatasetsFromDirectory(path: Union[str, Path]) -> Union[np.ndarray, bool]:
    """Load all datasets from a directory.

    Parameters
    ----------
    path : Union[str, Path]
        Path to directory containing datasets.

    Returns
    -------
    Union[np.ndarray, bool]
        Numpy array containing all datasets, or False if path is not a directory.
    """
    dirPath = Path(path)
    if not dirPath.is_dir():
        return False
    dataset = np.array([], dtype=object)
    for p in tqdm.tqdm(dirPath.glob("*.joblib"), desc="Loading datasets"):
        tmpDataset = load_dataset(p)
        dataset = np.append(dataset, tmpDataset, axis=0)
        # print(len(tmpDataset))
    return dataset


def save_trajectories(
    trajectories: Union[List, np.ndarray],
    output: Union[str, Path],
    classifier: str = "SVM",
    name: str = "trajectories",
) -> List[str]:
    """Save trajectories to a file.

    Parameters
    ----------
    trajectories : Union[List, np.ndarray]
        Trajectories to save.
    output : Union[str, Path]
        Output directory path.
    classifier : str, optional
        Name of classifier, by default "SVM"
    name : str, optional
        Additional name to identify file, by default "trajectories"

    Returns
    -------
    List[str]
        List of saved file paths.
    """
    _filename = Path(output) / f"{classifier}_{name}.joblib"
    return joblib.dump(trajectories, filename=_filename)


def load_dataset(path2dataset: Union[str, List[str], Path]) -> np.ndarray:
    """Load a dataset from a file or a directory.

    Parameters
    ----------
    path2dataset : Union[str, List[str], Path]


    Returns
    -------
    np.ndarray
        Numpy array containing the dataset.

    Raises
    ------
    IOError
        Wrong file type.
    """
    if type(path2dataset) == list:
        datasets = []
        for p in tqdm.tqdm(path2dataset):
            datasets.append(load_dataset(p))
        return mergeDatasets(datasets)
    datasetPath = Path(path2dataset)
    ext = datasetPath.suffix
    if ext == ".joblib":
        try:
            dataset = joblib.load(path2dataset)
        except Exception as e:
            print(f"Error loading {path2dataset}: {e}")
            return np.array([])
        if type(dataset[0]) == dict:
            ret_dataset = [d["track"] for d in dataset]
            dataset = ret_dataset
        for d in dataset:
            d._dataset = path2dataset
        return np.array(dataset)
    # elif ext == ".db":
    #     return np.array(preprocess_database_data_multiprocessed(path2dataset, n_jobs=None))
    elif Path.is_dir(datasetPath):
        return mergeDatasets(loadDatasetsFromDirectory(datasetPath))
    raise IOError("Wrong file type.")


def mergeDatasets(datasets: np.ndarray):
    """Merge multiple datasets into one.

    Parameters
    ----------
    datasets : np.ndarray
        List of datasets to merge.

    Returns
    -------
    np.ndarray
        Merged dataset.
    """
    merged = np.array([])
    for d in datasets:
        merged = np.append(merged, d)
    return merged
