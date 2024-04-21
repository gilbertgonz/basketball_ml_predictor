import numpy as np
import numpy.linalg as la
import cv2
import os
import math

def kalman(mu,P,z, fps):
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

    ####### Kalman filter params #######
    dt    = 1/fps
    noise = 3

    # Transition Matrix
    A = np.array(
        [1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1 ]).reshape(4,4)

    # Adjust a and B accordingly
    u = np.array([0, 9000])

    # Control Matrix
    B = np.array(
        [dt**2/2, 0,
        0, dt**2/2,
        dt, 0,
        0, dt ]).reshape(4,2)

    # Measurement Matrix
    H = np.array(
        [1,0,0,0,
        0,1,0,0]).reshape(2,4)
    
    sigmaM = 0.0001
    sigmaZ = 3*noise

    Q = sigmaM**2 * np.eye(4) # process noise cov
    R = sigmaZ**2 * np.eye(2) # measurement noise cov

    ####################################
    
    mup = A @ mu + B @ u
    pp  = A @ P @ A.T + Q

    zp = H @ mup

    # if no observation just do a prediction
    if z is None:
        return mup, pp, zp

    epsilon = z - zp

    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)

    new_mu = mup + k @ epsilon
    new_P  = (np.eye(len(P))-k @ H) @ pp
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
