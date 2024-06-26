import numpy as np
import numpy.linalg as la
import cv2
import os
import math

def kalman(mu, P, z, fps):
    """
        mu: Current state vector.
        P:  Covariance matrix.
        A:  Dynamic system matrix.
        Q:  Covariance matrix of the process noise.
        B:  Control model matrix.
        u:  Control input vector.
        z:  Observation vector.
        H:  Observation model matrix.
        R:  Covariance matrix of the observation noise.
    """

    ####### Kalman filter parameters #######
    dt = 1 / fps  # Time step
    noise = 3     # Measurement noise parameter

    # Transition Matrix: Defines how the state evolves over time
    A = np.array(
        [1, 0, dt, 0,
         0, 1, 0, dt,
         0, 0, 1, 0,
         0, 0, 0, 1]).reshape(4, 4)

    # Control input vector (e.g., forces or accelerations applied to system)
    u = np.array([0, 9000])

    # Control Matrix: Defines how control inputs affect state
    B = np.array(
        [dt**2 / 2, 0,
         0, dt**2 / 2,
         dt, 0,
         0, dt]).reshape(4, 2)

    # Measurement Matrix: Defines how the state is mapped to observation space
    H = np.array(
        [1, 0, 0, 0,
         0, 1, 0, 0]).reshape(2, 4)

    # Process noise covariance matrix
    sigmaM = 0.0001
    Q = sigmaM**2 * np.eye(4)

    # Measurement noise covariance matrix
    sigmaZ = 3 * noise
    R = sigmaZ**2 * np.eye(2)

    ####################################

    # Prediction Step
    mup = A @ mu + B @ u  # Predict next state based on current state and control input
    pp = A @ P @ A.T + Q  # Predict next covariance matrix based on current covariance and process noise

    # Predict observation based on predicted state
    zp = H @ mup

    # If no observation, return predicted state and covariance
    if z is None:
        return mup, pp, zp

    # Update Step
    epsilon = z - zp  # Compute residual between actual and predicted observation

    # Compute gain
    k = pp @ H.T @ la.inv(H @ pp @ H.T + R)

    # Update state estimate based on residual and gain
    new_mu = mup + k @ epsilon

    # Update covariance matrix based on gain and predicted covariance
    new_P = (np.eye(len(P)) - k @ H) @ pp

    return new_mu, new_P, zp

def detect(model, cv_image, data, thresh=0.55):
    results = model(cv_image, conf=thresh, verbose=False, max_det = 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            bbox = box.xyxy[0]
            x1, y1, x2, y2 = bbox
            c = box.cls

            if c == 0:
                data['ball'] += [(int((x1 + x2) / 2), int((y1 + y2) / 2))] # center of ball
            if c == 1:
                data['rim'] += [[(x1, y1), (x2, y1)]] # top-left and top-right of rim

    return results

def polyfit(frame, xpoints, ypoints, for_range, rim_y):
    A, B, C = np.polyfit(xpoints, ypoints, 2)
    for x in range(for_range[0], for_range[1]):
        y = int(A * x ** 2 + B * x + C)
        cv2.circle(frame,(x, y), 5, (0, 0, 255), -1)

    a = A
    b = B
    c = C - rim_y

    d = b*b - 4*a*c # discriminant
    if d >= 0:
        x1 = int((-b - math.sqrt(d)) / (2 * a)) # solution 1
        x2 = int((-b + math.sqrt(d)) / (2 * a)) # solution 2

        return x1, x2
    else:
        return None, None

def interpolate_x_for_y(x_list, y_list, y_target):
    if y_target in y_list:
        index = y_list.index(y_target)
        return x_list[index]
    else:
        # Find the closest y value in y_list to y_target
        closest_y = min(y_list, key=lambda y: abs(y - y_target))
        closest_index = y_list.index(closest_y)
        
        # Interpolate to estimate x value for y_target
        x1, x2 = x_list[closest_index], x_list[closest_index + 1]
        y1, y2 = y_list[closest_index], y_list[closest_index + 1]
        
        x_estimate = x1 + (x2 - x1) * (y_target - y1) / (y2 - y1)
        return x_estimate
    
def save_images(path, img, counter):
    os.makedirs(path, exist_ok=True)
    filename = f"{path}/img_{counter:04d}.jpg"
    cv2.imwrite(filename, img)
