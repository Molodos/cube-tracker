# Some code is from here: https://aliyasineser.medium.com/aruco-marker-tracking-with-opencv-8cb844c26628

import cv2
import cv2.aruco as aruco
import numpy as np

import calibrate_camera

cap = cv2.VideoCapture(0)  # Get the camera source


def track(matrix_coefficients, distortion_coefficients):
    while True:
        ret, frame = cap.read()
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
            for i in range(0, len(ids)):  # Iterate in markers
                marker_id = ids[i]
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                success, rvec, tvec = cv2.solvePnP(marker_points, corners[i], matrix_coefficients,
                                                   distortion_coefficients)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 30)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_matrix, dist_matrix = calibrate_camera.load_coefficients("camera.yml")
    track(camera_matrix, dist_matrix)
