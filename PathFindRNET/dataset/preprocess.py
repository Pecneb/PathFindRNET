import numpy as np
import tqdm
from itertools import filterfalse


def euclidean_distance(q1: float, p1: float, q2: float, p2: float):
    """
    Calculate the Euclidean distance between two 2D vectors.

    Parameters
    ----------
    q1 : float
        First element of vector1.
    p1 : float
        First element of vector2.
    q2 : float
        Second element of vector1.
    p2 : float
        Second element of vector2.

    Returns
    -------
    float
        The Euclidean distance between the two vectors.
    """
    return (((q1 - p1) ** 2) + ((q2 - p2) ** 2)) ** 0.5


def filter_out_false_positive_detections_by_enter_exit_distance(
    trackedObjects, threshold
):
    """
    Filter out false positive detections based on the distance between their enter and exit points.

    Parameters
    ----------
    trackedObjects : list
        List of object trackings.
    threshold : float
        If the distance between the enter and exit points is closer than this threshold value,
        the detections are excluded from the returned list.

    Returns
    -------
    list
        List of filtered tracks.

    Notes
    -----
    This function filters out false positive detections by calculating the Euclidean distance
    between the enter and exit points of each tracked object. If the distance is less than the
    specified threshold, the detection is considered a false positive and excluded from the
    returned list of tracks.
    """
    filteredTracks = list(
        filterfalse(
            lambda obj: euclidean_distance(
                obj.history[0].X, obj.history[-1].X, obj.history[0].Y, obj.history[-1].Y
            )
            < threshold,
            trackedObjects,
        )
    )
    return filteredTracks


def is_noise(trackedObject, threshold: float = 0.1):
    """
    Check if the tracked object's trajectory contains noise.
    This is done by calculating the Euclidean distance between each point in the trajectory.
    If two consecutive points are further apart than the specified threshold value, the trajectory
    is considered to contain noise.

    Parameters
    ----------
    trackedObject : object
        The tracked object containing the trajectory information.
    threshold : float, optional
        The threshold value to determine if a movement is considered noise. Defaults to 0.1.

    Returns
    -------
    bool
        True if the trajectory contains noise, False otherwise.
    """
    for i in range(len(trackedObject.history_X) - 1):
        # d = np.linalg.norm(np.array([trackedObject.history_X[i], trackedObject.history_Y[i]]) -
        #                    np.array([trackedObject.history_X[i+1], trackedObject.history_Y[i+1]]))
        d = euclidean_distance(
            trackedObject.history_X[i],
            trackedObject.history_X[i + 1],
            trackedObject.history_Y[i],
            trackedObject.history_Y[i + 1],
        )
        if d > threshold:
            return True
    return False


def filter_out_noise_trajectories(trackedObjects: list, distance: float = 0.05):
    """
    Filter out noise trajectories from a list of tracked objects.

    Parameters
    ----------
    trackedObjects : list
        List of tracked objects.
    distance : float, optional
        Maximum distance threshold to consider an object as noise. Defaults to 0.05.

    Returns
    -------
    numpy.ndarray
        Filtered array of tracked objects without noise.
    """
    filtered = []
    for i in range(len(trackedObjects)):
        if not is_noise(trackedObjects[i], distance):
            filtered.append(trackedObjects[i])
    return np.array(filtered)


def search_min_max_coordinates(trackedObjects):
    """
    Search for minimum and maximum of x,y coordinates,
    to find the edges of the range of interest.

    Parameters
    ----------
    trackedObjects : list
        List of trajectory objects.

    Returns
    -------
    tuple
        Tuple consisting of min x,y and max x,y coordinates.
    """
    X = np.concatenate([o.history_X for o in trackedObjects], axis=None)
    Y = np.concatenate([o.history_Y for o in trackedObjects], axis=None)
    min_x = np.min(X)
    max_x = np.max(X)
    min_y = np.min(Y)
    max_y = np.max(Y)
    return min_x, min_y, max_x, max_y


def filter_out_edge_detections(trackedObjects, threshold):
    """
    Filter out objects based on their entering and exiting detection coordinates.

    Parameters
    ----------
    trackedObjects : list
        List of object trackings.
    threshold : float
        Threshold value for filtering. Only objects under this value will be returned.

    Returns
    -------
    list
        Filtered list of tracks.
    """
    min_x, min_y, max_x, max_y = search_min_max_coordinates(trackedObjects)
    filteredTracks = []
    for obj in tqdm.tqdm(trackedObjects, desc="Filter out edge detections."):
        if (
            (
                obj.history[0].X <= min_x + threshold
                or obj.history[0].X >= max_x - threshold
            )
            or (
                obj.history[0].Y <= min_y + threshold
                or obj.history[0].Y >= max_y - threshold
            )
        ) and (
            (
                obj.history[-1].X <= min_x + threshold
                or obj.history[-1].X >= max_x - threshold
            )
            or (
                obj.history[-1].Y <= min_y + threshold
                or obj.history[-1].Y >= max_y - threshold
            )
        ):
            filteredTracks.append(obj)
    return filteredTracks


def filter_trajectories(
    trackedObjects,
    threshold=0.7,
    enter_exit_dist=0.4,
    detectionDistanceFiltering=True,
    detDist=0.05,
):
    """
    Run filtering process on trajectory dataset.

    Parameters
    ----------
    trackedObjects : list
        List of tracked objects representing trajectories.
    threshold : float, optional
        Threshold for min max search. Defaults to 0.7.
    enter_exit_dist : float, optional
        Distance threshold for filtering false positive detections based on enter/exit distance. Defaults to 0.4.
    detectionDistanceFiltering : bool, optional
        Flag to enable/disable detection distance filtering. Defaults to True.
    detDist : float, optional
        Distance threshold for filtering noise trajectories. Defaults to 0.05.

    Returns
    -------
    numpy.ndarray
        Filtered trajectories without noise.
    """
    edge_trajectories = filter_out_edge_detections(trackedObjects, threshold)
    filtered_trajectories = filter_out_false_positive_detections_by_enter_exit_distance(
        edge_trajectories, enter_exit_dist
    )
    if detectionDistanceFiltering:
        denoised_trajectories = filter_out_noise_trajectories(
            filtered_trajectories, detDist
        )
        return np.array(denoised_trajectories, dtype=object)
    return np.array(filtered_trajectories, dtype=object)


def filter_by_class(trackedObjects: list, label="car"):
    """
    Only return objects with the given label.

    Parameters
    ----------
    trackedObjects : list
        List of tracked objects.
    label : str, optional
        Label of the object. Defaults to "car".

    Returns
    -------
    list
        List of objects with the specified label.
    """
    return [obj for obj in trackedObjects if obj.label == label]
