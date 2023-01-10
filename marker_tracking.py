import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse
import pandas as pd 
from pykalman import KalmanFilter
import time
import sys, math

cap = cv2.VideoCapture('Ref4_S3.mp4')

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
data = {'Marker': [], 'x': [], 'y': [], 'z': [], 't (s)': [], 'Frame': []}
df = pd.DataFrame(data)
frame_count = 0
initial_state_mean = np.empty((0,4))  

firstMarkerID = None
secondMarkerID = None

#--- 180 deg rotation matrix around the x axis
R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def t_get(frame_count, fps):
    t = frame_count*(1/fps)
    return t

def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    
    dirpath = 'CalibrationImages/'; 
    prefix = 'IMG'; image_format = 'jpg';
    images = glob.glob(dirpath+'/' + prefix + '*.' + image_format)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return [ret, mtx, dist, rvecs, tvecs]


def saveCoefficients(mtx, dist):
    cv_file = cv2.FileStorage("CalibrationImages/calibrationCoefficients.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def loadCoefficients():
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage("CalibrationImages/calibrationCoefficients.yaml", cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    # Debug: print the values
    # print("camera_matrix : ", camera_matrix.tolist())
    # print("dist_matrix : ", dist_matrix.tolist())

    cv_file.release()
    return [camera_matrix, dist_matrix]


def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    return img


def initial_kalman(transition_matrix, observation_matrix, initial_state_mean, measurements, observation_covariance = None, kf = None):
    if type(observation_covariance) == type(None) and type(kf) == type(None):
        kf = KalmanFilter(transition_matrices = transition_matrix,
                      observation_matrices = observation_matrix,
                      initial_state_mean = initial_state_mean)
        
        kf = kf.em(measurements, n_iter=5)
        filtered_state_means, filtered_state_covariances = kf.smooth(measurements)

    else: 
        kf = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean,
                  observation_covariance = 10*kf.observation_covariance,
                  em_vars=['transition_covariance', 'initial_state_covariance'])
        
        kf = kf.em(measurements, n_iter=5)
        filtered_state_means, filtered_state_covariances = kf.filter(measurements)
    return kf, filtered_state_means, filtered_state_covariances


def update_kalman(kf, now, P_now, new, measurement):
    now, P_now = kf.filter_update(filtered_state_mean = now,
                                   filtered_state_covariance = P_now,
                                   observation = measurement)
    new = np.vstack([new, now])
    return now, P_now, new
    

def track(matrix_coefficients, distortion_coefficients, df, frame_count, initial_state_mean, first_run):
    pointCircle = (0, 0)
    markerTvecList = []
    markerRvecList = [] 
    
    cam_top_left = []
    cam_top_right = []
    cam_bot_left = []
    cam_bot_right = []
    
    composedRvec, composedTvec = None, None
    
    while True:
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        t = t_get(frame_count, fps)
        frame_count += 1
        
        ret, frame = cap.read() #splices video into frames and reads the frames 
        frame = ResizeWithAspectRatio(frame, width=1280)
        
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)  # Use 4x4 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
                                                                #output coordinates of the 4 corners and the ID of the marker 

        if np.all(ids is not None):  # If there are markers found by detector
            del markerTvecList[:]
            del markerRvecList[:]
            zipped = zip(ids, corners)
            ids, corners = zip(*(sorted(zipped)))
            axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.04, matrix_coefficients, distortion_coefficients)
                #40mm = 0.04m
                #pull out the roation of the marker and the tvec is the center of the four corners

                if ids[i] == firstMarkerID:
                    firstRvec = rvec
                    firstTvec = tvec
                    isFirstMarkerCalibrated = True
                    firstMarkerCorners = corners[i]
                elif ids[i] == secondMarkerID:
                    secondRvec = rvec
                    secondTvec = tvec
                    isSecondMarkerCalibrated = True
                    secondMarkerCorners = corners[i]

                
                # Define Reference IDs
                top_left = 30
                top_right = 20
                bot_left = 40
                bot_right = 0
                moving_marker = 10
                
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                markerRvecList.append(rvec)
                markerTvecList.append(tvec)

                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                
                cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.03)
                
             
                x = (corners[i-1][0][0][0] + corners[i-1][0][1][0] + corners[i-1][0][2][0] + corners[i-1][0][3][0]) / 4

                y = (corners[i-1][0][0][1] + corners[i-1][0][1][1] + corners[i-1][0][2][1] + corners[i-1][0][3][1]) / 4
                
                
                # Prepare Kalman filter parameters
                # Assume no acceleration
                
                # Eqns: 
                #x(k) = x(k-1) + dt*x_dot(k-1); [1 1 0 0]
                #x_dot(k) = x_dot(k-1); [0 1 0 0]
                #y(k) = y(k-1) + dt*y_dot(k-1); [0 0 1 1]
                #y_dot(k) = y_dot(k-1); [0 0 0 1]

                #rows: x, vx, y, vy, z, vz
                #columns: x, vx, y, vy
                transition_matrix = [[1, 1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0],
                                 [0, 0, 1, 1, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 1],
                                 [0, 0, 0, 0, 0, 1]]
                
                # We only observe x and y, not their velocities
                # columns: x, vx, y, vy, z, vz
                # rows: x, y, z 
                observation_matrix = [[1, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0]]

                x_kalman = tvec[0][0][0]
                y_kalman = tvec[0][0][1]
                z_kalman = tvec[0][0][2]

                if first_run:
                    first_run = False
                # If not initial run
                if not first_run:
                    
                    # Implement Kalman Filter
                    if not np.any(initial_state_mean) and ids[i][0] == moving_marker:
                        
                        # Initialize Kalman Filter 
                        measurements = np.column_stack(([x_kalman, x], [y_kalman, y], [z_kalman, z]))
                        
                        initial_state_mean =  np.array((x, 0, y, 0, z, 0))
                        kf1, filtered_state_means, filtered_state_covariances = initial_kalman(transition_matrix, observation_matrix, initial_state_mean, measurements)

                        # Add smoothening function
                        kf2, filtered_state_means, filtered_state_covariances = initial_kalman(transition_matrix, observation_matrix, initial_state_mean, measurements, observation_covariance = 10*kf1.observation_covariance, kf = kf1)
                        
                        now = filtered_state_means[-1, :]
                        P_now = filtered_state_covariances[-1, :]
                        new = np.zeros(6)
                        
                        
                    # Implemented Updated Filter  
                    if ids[i][0] == moving_marker:
                        measurements = np.append(measurements,[[x,y,z]], axis=0)
                        measurement = np.array([x,y,z])
                        
                        now, P_now, new = update_kalman(kf2, now, P_now, new, measurement)
                        
                        x_kalman = new[-1,0]
                        y_kalman = new[-1,2]
                        z_kalman = new[-1,4]
                        
                x = x_kalman
                y = y_kalman
                z = z_kalman
                
                df2 = pd.DataFrame([[ids[i][0], x, y, z, t, frame_count]], columns=['Marker', 'x', 'y', 'z', 't (s)', 'Frame'])
                df = pd.concat([df, df2])

            if len(ids) > 1 and composedRvec is not None and composedTvec is not None:
                info = cv2.composeRT(composedRvec, composedTvec, secondRvec.T, secondTvec.T)
                TcomposedRvec, TcomposedTvec = info[0], info[1]

                objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
                imgpts, jac = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, matrix_coefficients,
                                                distortion_coefficients)

                # frame = draw(frame, corners[0], imgpts)
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec,
                               0.01)  # Draw Axis
                relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
                cv2.circle(frame, relativePoint, 2, (255, 255, 0))

    
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):  # Quit
            
            # Delete this if you want all markers
            #df = df[df['Marker'] == 0]
            df = df.iloc[1:]
            
            df.to_csv('3d_mapping_old.csv', index=False)
            
            break
        elif key == ord('c'):  # Calibration
            if len(ids) > 1:  # If there are two markers, reverse the second and get the difference
                firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
                secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))

                composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)
                
    return df

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aruco Marker Tracking')
    parser.add_argument('--coefficients', metavar='bool', required=True,
                        help='File name for matrix coefficients and distortion coefficients')
    parser.add_argument('--firstMarker', metavar='int', required=True,
                        help='Marker ID for the first marker')
    parser.add_argument('--secondMarker', metavar='int', required=True,
                        help='Marker ID for the second marker')


    # Parse the arguments and take action for that.
    args = parser.parse_args()
    firstMarkerID = int(args.firstMarker)
    secondMarkerID = int(args.secondMarker)

    if args.coefficients == '1':
        mtx, dist = loadCoefficients()
        ret = True
    else:
        ret, mtx, dist, rvecs, tvecs = calibrate()
        saveCoefficients(mtx, dist)
    print("Calibration is completed. Starting tracking sequence.")
    

    if ret:
        first_run = True
        df = track(mtx, dist, df, frame_count, initial_state_mean, first_run)