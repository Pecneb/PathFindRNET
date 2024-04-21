"""
    Predicting trajectories of objects
    Copyright (C) 2022  Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import tqdm


@dataclass
class Detection:
    """
    A class representing a detection in a video frame.

    Attributes
    ----------
    label : str
        The label of the detected object.
    confidence : float
        The confidence score of the detection.
    X : float
        The x-coordinate of the top-left corner of the bounding box.
    Y : float
        The y-coordinate of the top-left corner of the bounding box.
    Width : float
        The width of the bounding box.
    Height : float
        The height of the bounding box.
    frameID : int
        The ID of the frame in which the detection was made.
    VX : float, optional
        The x-velocity of the detected object. Default is None.
    VY : float, optional
        The y-velocity of the detected object. Default is None.
    AX : float, optional
        The x-acceleration of the detected object. Default is None.
    AY : float, optional
        The y-acceleration of the detected object. Default is None.
    objID : int, optional
        The ID of the detected object. Default is None.

    Methods
    -------
    __repr__() -> str
        Returns a string representation of the Detection object.
    __eq__(other) -> bool
        Returns True if the given Detection object is equal to this Detection object, False otherwise.
    """

    label: str
    confidence: float
    X: float
    Y: float
    Width: float
    Height: float
    frameID: int
    VX: float = field(
        init=False,
    )
    VY: float = field(init=False)
    AX: float = field(init=False)
    AY: float = field(init=False)
    objID: int = field(init=False)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Detection object.

        Returns
        -------
        str
            A string representation of the Detection object.
        """
        return f"Label: {self.label}, Confidence: {self.confidence}, X: {self.X}, Y: {self.Y}, Width: {self.Width}, Height: {self.Height}, Framenumber: {self.frameID}"

    def __eq__(self, other) -> bool:
        """
        Returns True if the given Detection object is equal to this Detection object, False otherwise.

        Parameters
        ----------
        other : Detection
            The Detection object to compare with.

        Returns
        -------
        bool
            True if the given Detection object is equal to this Detection object, False otherwise.
        """
        if self.label != other.label:
            return False
        if self.confidence != other.confidence:
            return False
        if self.X != other.X:
            return False
        if self.Y != other.Y:
            return False
        if self.Width != other.Width:
            return False
        if self.Height != other.Height:
            return False
        if self.frameID != other.frameID:
            return False
        return True


@dataclass
class TrackedObject:
    """This class represents a tracked object.

    Parameters
    ----------
    id : int
        Object ID.
    first : Detection
        First Detection object.
    max_age : int, optional
        max_age is an argument which defines the tracking mechanism's memory
        how far can we re-detect, by default 30

    Attributes
    ----------
    objID : int
        Object ID.
    label : int
        Object label.
    history_X : np.ndarray
        History of x-coordinates.
    history_Y : np.ndarray
        History of y-coordinates.
    history_VX_kalman : np.ndarray
        History of x-velocities.
    history_VY_kalman : np.ndarray
        History of y-velocities.
    isMoving : bool
        True if the object is moving, False otherwise.
    time_since_update : int
        Time since last update.
    max_age : int
        Maximum age of the object.
    mean : list
        Object values calculated by Kalman filter.
    dataset : str
        Dataset name.
    offline : bool
        True if the object is offline, False otherwise.

    Methods
    -------
    __init__(id: int, first: Detection, max_age: int = 30)
        Initializes the TrackedObject object.
    from_detections(id: int, detections: List[Detection], max_age: int = 30)
        Creates a TrackedObject instance from a list of Detection objects.
    __repr__() -> str
        Returns a string representation of the TrackedObject object.
    __hash__() -> int
        Returns the hash value of the TrackedObject object.
    __eq__(other) -> bool
        Returns True if the given TrackedObject object is equal to this TrackedObject object, False otherwise.
    update(detection: Detection = None, mean=None)
        Updates the tracked object state with new detection.
    """

    objID: int
    label: int = field(init=False)
    history: List[Detection] = field(init=False)
    history_X: np.ndarray = field(init=False)
    history_Y: np.ndarray = field(init=False)
    history_VX_kalman: np.ndarray = field(init=False)
    history_VY_kalman: np.ndarray = field(init=False)
    isMoving: bool = field(init=False)
    time_since_update: int = field(init=False)
    max_age: int
    mean: list = field(init=False)
    dataset: str = field(init=False)
    offline: bool = field(init=False)

    def __init__(self, id: int, first: Detection, max_age: int = 30):
        self.objID = id
        self.history = [first]
        self.history_X = np.array([first.X])
        self.history_Y = np.array([first.Y])
        self.history_VX_kalman = np.array([])
        self.history_VY_kalman = np.array([])
        self.label = first.label
        self.isMoving = False
        self.max_age = max_age
        self.time_since_update = 0
        self.mean = []
        self.dataset = ""
        self.offline = False

    @staticmethod
    def from_detections(id: int, detections: List[Detection], max_age: int = 30):
        """Create a TrackedObject instance from a list of Detection objects.

        Parameters
        ----------
        id : int
            Object ID.
        detections : List[Detection]
            List of Detection objects.
        max_age : int, optional
            max_age is an argument which defines the tracking mechanism's memory
            how far can we re-detect, by default 30
        """
        tmpObj = TrackedObject(
            id,
            Detection(
                detections[0].label,
                detections[0].confidence,
                detections[0].X,
                detections[0].Y,
                detections[0].Width,
                detections[0].Height,
                detections[0].frameID,
            ),
            max_age,
        )
        for d in detections:
            tmpObj.history.append(
                Detection(d.label, d.confidence, d.X, d.Y, d.Width, d.Height, d.frameID)
            )
            tmpObj.history_X = np.append(tmpObj.history_X, [d.X])
            tmpObj.history_Y = np.append(tmpObj.history_Y, [d.Y])
            # tmpObj.history_VX_kalman = np.append(tmpObj.history_VX_kalman, [d.VX])
            # tmpObj.history_VY_kalman = np.append(tmpObj.history_VY_kalman, [d.VY])
        return tmpObj

    def __repr__(self) -> str:
        return f"ID: {self.objID}, Label: {self.label}"

    def __hash__(self) -> int:
        retval = int(
            self.objID
            + np.sum([self.history[i].frameID for i in range(len(self.history))])
        )
        # print(retval, self.objID, self._dataset)
        return retval

    def __eq__(self, other: "TrackedObject") -> bool:
        for i in range(len(self.history)):
            for j in range(len(other.history)):
                if self.history[i] != other.history[j]:
                    return False
        return self.objID == other.objID

    def update(self, detection: Detection = None, mean=None):
        """Update the tracked object state with new detection.

        Parameters
        ----------
        detection : Detection, optional
            New Detection from yolo, by default None
        mean : List, optional
            Object values calculated by Kalman filter, by default None
        """
        if detection is not None:
            self.history.append(detection)
            self.history_X = np.append(self.history_X, [detection.X])
            self.history_Y = np.append(self.history_Y, [detection.Y])
            self.history_VX_kalman = np.append(self.history_VX_kalman, mean[4])
            self.history_VY_kalman = np.append(self.history_VY_kalman, mean[5])
            self.time_since_update = 0
        else:
            self.time_since_update += 1
        if (
            np.abs(self.history_VX_kalman[-1]) > 0.0
            or np.abs(self.history_VY_kalman[-1]) > 0.0
        ) and len(self.history_X) >= 5:
            self.isMoving = (
                (self.history_X[-5] - self.history_X[-1]) ** 2
                + (self.history_Y[-5] - self.history_Y[-1]) ** 2
            ) ** (1 / 2) > 5.0
        else:
            self.isMoving = False
