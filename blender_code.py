# Some code is from here: https://aliyasineser.medium.com/aruco-marker-tracking-with-opencv-8cb844c26628
""" (Blender prefix)
import sys

package_path = "C:/Users/micha/AppData/Roaming/Python/Python310/site-packages/cv2"
sys.path.append(package_path)

import bpy


def findArea():
    for a in bpy.data.window_managers[0].windows[0].screen.areas:
        if a.type == "VIEW_3D":
            return a
    return None

obj = bpy.context.selected_objects[0]
r3d = findArea().spaces[0].region_3d

#Apply final matrix to object
def handleObject(matrix):
    obj.matrix_world = matrix

#Apply final matrix to viewport
def handleViewport(matrix):
    if area is None:
        print("area not found")
    else:
        # print(dir(area))
        r3d = area.spaces[0].region_3d
        r3d.view_matrix = matrix
"""


import math
from threading import Thread

import cv2
import cv2.aruco as aruco
import numpy
import numpy as np


def handleObject(matrix):
    pass


class CubeTracker(Thread):

    def __init__(self, camera_matrix, dist_matrix):
        super().__init__()
        super().setDaemon(True)
        self.cap = cv2.VideoCapture(0)  # Get the camera source
        self.camera_matrix = camera_matrix
        self.dist_matrix = dist_matrix

    @classmethod
    def get_rot_x(self, rad: float) -> numpy.ndarray:
        """
        Rotation matrix around x-axis
        """
        return numpy.transpose(numpy.asarray([
            [1, 0, 0, 0],
            [0, math.cos(rad), math.sin(rad), 0],
            [0, -math.sin(rad), math.cos(rad), 0],
            [0, 0, 0, 1]
        ]))

    @classmethod
    def get_rot_y(self, rad: float) -> numpy.ndarray:
        """
        Rotation matrix around y-axis
        """
        return numpy.transpose(numpy.asarray([
            [math.cos(rad), 0, -math.sin(rad), 0],
            [0, 1, 0, 0],
            [math.sin(rad), 0, math.cos(rad), 0],
            [0, 0, 0, 1]
        ]))

    @classmethod
    def get_rot_z(self, rad: float) -> numpy.ndarray:
        """
        Rotation matrix around z-axis
        """
        return numpy.transpose(numpy.asarray([
            [math.cos(rad), -math.sin(rad), 0, 0],
            [math.sin(rad), math.cos(rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]))

    def get_cube_4x4(self, marker_id: int, rvec) -> numpy.ndarray:
        """
        Beautiful code
        """
        # Basic stuff
        rmat = cv2.Rodrigues(rvec)[0]
        rmat = numpy.transpose(rmat)

        # Generate 4x4 matrix
        four_mat: list[list[float]] = []
        for col in rmat:
            four_mat.append([float(i) for i in col] + [0])
        four_mat.append([0, 0, 0, 1])

        # Rotate sides different from 4
        if marker_id == 0:
            four_mat = numpy.matmul(self.get_rot_x(math.pi / 2), four_mat)
        elif marker_id == 1:
            four_mat = numpy.matmul(self.get_rot_x(math.pi / 2), four_mat)
            four_mat = numpy.matmul(self.get_rot_z(-math.pi / 2), four_mat)
        elif marker_id == 2:
            four_mat = numpy.matmul(self.get_rot_x(math.pi / 2), four_mat)
            four_mat = numpy.matmul(self.get_rot_z(math.pi), four_mat)
        elif marker_id == 3:
            four_mat = numpy.matmul(self.get_rot_x(math.pi / 2), four_mat)
            four_mat = numpy.matmul(self.get_rot_z(math.pi / 2), four_mat)
        elif marker_id == 4:
            pass
        elif marker_id == 5:
            four_mat = numpy.matmul(self.get_rot_x(math.pi), four_mat)

        return four_mat

    def run(self):
        while True:
            ret, frame = self.cap.read()
            # operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # Use 4x4 dictionary to find markers
            parameters = aruco.DetectorParameters()  # Marker detection parameters
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            # lists of ids and the corners beloning to each id
            marker_size = 50
            marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                                      [marker_size / 2, marker_size / 2, 0],
                                      [marker_size / 2, -marker_size / 2, 0],
                                      [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
            corners, ids, rejected_img_points = detector.detectMarkers(gray)

            if np.all(ids is not None):  # If there are markers found by detector
                # Got through all found markers
                cube_4x4_mats = []
                t_vecs = []

                for i in range(0, len(ids)):
                    # Get id
                    marker_id: int = ids[i][0]
                    # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                    success, rvec, tvec = cv2.solvePnP(marker_points, corners[i], self.camera_matrix,
                                                       self.dist_matrix)
                    mat = self.get_cube_4x4(marker_id, rvec)
                    cube_4x4_mats.append(mat)
                    t_vecs.append(numpy.asarray([i[0] for i in tvec]))

                summ = numpy.asarray([
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]
                ])
                i = 0
                for mat in cube_4x4_mats:
                    i += 1
                    summ = numpy.add(summ, mat)
                average = numpy.divide(summ, i)

                # TODO: add translation (-30?)
                summ_tvec = numpy.asarray([0, 0, 0])
                i = 0
                for tvec in t_vecs:
                    i += 1
                    summ_tvec = numpy.add(summ_tvec, tvec)
                average_tvec = numpy.divide(summ_tvec, i)
                trans = numpy.transpose(numpy.asarray([
                    [0, 0, 0, average_tvec[0]],
                    [0, 0, 0, average_tvec[1]],
                    [0, 0, 0, average_tvec[2]],
                    [0, 0, 0, 0]
                ]))
                average = numpy.add(average, trans)

                if i > 0:
                    handleObject(average)


camera_matrix = numpy.asarray([[1.4382180749827194e+03, 0., 9.7152746163427696e+02], [0.,
                                                                                      1.4367159888379267e+03,
                                                                                      5.8118828164621232e+02],
                               [0., 0., 1.]])
dist_matrix = numpy.asarray([1.2230913272997351e-01, -1.0362727004218768e-01,
                             7.4754497718445816e-03, 1.1768358506292884e-02,
                             -6.3270009873691399e-02])
cube_tracker = CubeTracker(camera_matrix, dist_matrix)
cube_tracker.start()
