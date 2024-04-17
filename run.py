#!/usr/bin/env python3

import numpy as np
import cv2
import warnings
import torch
from ultralytics import YOLO
import argparse

from utils import * 

kalman_predict = True
polynomial_predict = True
img_counter = 0

def main(vidpath):
    global img_counter

    ####### Kalman filter params #######
    fps   = 150
    dt    = 1/fps
    noise = 3

    # Transition Matrix
    A = np.array(
        [1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1 ]).reshape(4,4)

    # Adjust a and B accordingly
    a = np.array([0, 9000])

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

    mu  = np.zeros(4) # x, y, vx, vy
    P   = np.diag([1000,1000,1000,1000])**2
    res =[]

    sigmaM = 0.0001
    sigmaZ = 3*noise

    Q = sigmaM**2 * np.eye(4) # process noise cov
    R = sigmaZ**2 * np.eye(2) # measurement noise cov

    ####################################
    data = {'ball':[],
            'rim':[]}

    ball_x_list = []
    ball_y_list = []

    prev_ball_x = 0
    step_counter = 0

    vid = cv2.VideoCapture(vidpath)
    
    while True:
        ret, frame = vid.read()

        if ret:
            # Detect and return annotated frame
            results = detect(model, frame, data)
            annotated_frame = results[0].plot()

            # Kalman legend
            cv2.rectangle(annotated_frame, (int(annotated_frame.shape[1] - 330), int(annotated_frame.shape[0] - 170)), (int(annotated_frame.shape[1] - 280), int(annotated_frame.shape[0] - 130)), (255, 0, 255), -1)
            cv2.putText(annotated_frame, "Kalman:", (int(annotated_frame.shape[1] - 270), int(annotated_frame.shape[0] - 140)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

            # Polynomial legend
            cv2.rectangle(annotated_frame, (int(annotated_frame.shape[1] - 330), int(annotated_frame.shape[0] - 120)), (int(annotated_frame.shape[1] - 280), int(annotated_frame.shape[0] - 80)), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "Poly:", (int(annotated_frame.shape[1] - 270), int(annotated_frame.shape[0] - 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Prediction legend
            cv2.putText(annotated_frame, "Prediction:", (int(annotated_frame.shape[1] - 330), int(annotated_frame.shape[0] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated_frame, "Prediction:", (int(annotated_frame.shape[1] - 330), int(annotated_frame.shape[0] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if len(data['ball']) > 0:
                ball_x = data['ball'][-1][0]
                ball_y = data['ball'][-1][1]
                

                ball_x_list.append(ball_x)
                ball_y_list.append(ball_y)

                if len(data['rim']) > 0:
                    rim_x1 = data['rim'][-1][0][0]
                    rim_x2 = data['rim'][-1][1][0]
                    rim_y  = data['rim'][-1][0][1]
                    rim_center = (rim_x2 + rim_x1) / 2

                    # Rim points
                    cv2.circle(annotated_frame,(int(rim_x1), int(rim_y)), 5, (0, 255, 0), 5)
                    cv2.circle(annotated_frame,(int(rim_x2), int(rim_y)), 5, (0, 255, 0), 5)
                else:
                    rim_y = 0

                # Calculating likelihood of successful basket based on ball destination point relative to rim
                if prev_ball_x != ball_x:
                    prediction_p = 0
                prediction_k = 0
                prediction_scale_factor = 5 # dampening prediction values
                
                # Polynomial regression
                if polynomial_predict:
                    if ball_x > prev_ball_x:
                        _, x2 = polyfit(annotated_frame, ball_x_list, ball_y_list, (ball_x, frame.shape[1]), rim_y)

                        if len(data['rim']) > 0 and x2 is not None:
                            # Ball trajectory intersection with rim_y
                            cv2.circle(annotated_frame,(x2, int(rim_y)), 5, (255, 255, 0), 5)

                            prediction_p += 50 - abs(x2 - rim_center) / prediction_scale_factor

                            # Check if est pos of ball is between rim corners
                            if int(rim_x1) < x2 < int(rim_x2):
                                cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 190), int(annotated_frame.shape[0] - 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    if ball_x < prev_ball_x:
                        x1, _ = polyfit(annotated_frame, ball_x_list, ball_y_list, (0, ball_x), rim_y)

                        if len(data['rim']) > 0 and x1 is not None:
                            cv2.circle(annotated_frame,(x1, int(rim_y)), 5, (255, 255, 0), 5)

                            prediction_p += 50 - abs(x1 - rim_center) / prediction_scale_factor

                            # Check if est pos of ball is between rim corners
                            if int(rim_x1) < x1 < int(rim_x2):
                                cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 190), int(annotated_frame.shape[0] - 90)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Kalman filter
                if kalman_predict:
                    # Initial prediction
                    mu, P, pred = kalman(mu, P, A, Q, B, a, np.array([ball_x, ball_y]), H, R)
                    res += [(mu, P)]

                    mu2  = mu
                    P2   = P
                    res2 = []

                    # Loop to predict entire trajectory
                    for _ in range(fps*2):
                        mu2, P2, pred2 = kalman(mu2, P2, A, Q, B, a, None, H, R)
                        res2 += [(mu2, P2)]
                    
                    # Predictions
                    xp = [mu2[0] for mu2, _ in res2]
                    yp = [mu2[1] for mu2, _ in res2]

                    # Standard deviations
                    xpu = [2 * np.sqrt(P[0, 0]) for _, P in res2]
                    ypu = [2 * np.sqrt(P[1, 1]) for _, P in res2]

                    # Draw predicted trajectory
                    for n in range(len(xp)):
                        if args.show_uncertainty:
                            uncertainty = (xpu[n] + ypu[n]) / 2
                            cv2.circle(annotated_frame, (int(xp[n]), int(yp[n])), int(uncertainty), (255, 0, 255))
                        else:
                            cv2.circle(annotated_frame,(int(xp[n]), int(yp[n])), 5, (255, 0, 255), -1)

                    if len(data['rim']) > 0:
                        # Find estimated x value of trajectory-rim intersection
                        xest = interpolate_x_for_y(xp, yp, rim_y)
                        cv2.circle(annotated_frame,(int(xest), int(rim_y)), 5, (255, 255, 0), 5)

                        prediction_k += 50 - abs(xest - rim_center) / prediction_scale_factor

                        # Check if est pos of ball is between rim corners
                        if int(rim_x1) < xest < int(rim_x2):
                                cv2.putText(annotated_frame, "Basket", (int(annotated_frame.shape[1] - 140), int(annotated_frame.shape[0] - 140)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Updating prev_ball_x to every 5th step bcuz steps are too small
                if step_counter % 5 == 0:
                    prev_ball_x = ball_x
            else:
                continue

            # Print prediction percentage
            prediction_total = round(min(max(float(prediction_k + prediction_p), 0), 100), 1)
            cv2.putText(annotated_frame, f"{prediction_total}%", (int(annotated_frame.shape[1] - 150), int(annotated_frame.shape[0] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"{prediction_total}%", (int(annotated_frame.shape[1] - 150), int(annotated_frame.shape[0] - 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow("Result", annotated_frame)
            step_counter += 1

            if save_path is not None:
                save_images(save_path, annotated_frame, img_counter)
                img_counter += 1

            # key = cv2.waitKey(0) & 0xFF # show per frame
            key = cv2.waitKey(1)
            if key == ord(" "):  # Spacebar to pause
               cv2.waitKey(-1)

        else:
            break

    vid.release()
    cv2.destroyAllWindows()   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-uncertainty", action="store_true", default=False, help="Set this flag show uncertainty from Kalman filter prediction")
    args = parser.parse_args()

    model = YOLO('best.pt')

    save_path = None # "imgs"

    # Set GPU if available
    if torch.cuda.is_available():
        model.to(device=torch.device("cuda"))
    else:
        model.to(device=torch.device("cpu")) 

    warnings.filterwarnings("ignore", category=np.RankWarning)  # suppressing this RankWarning since there may not 
                                                                # be enough points initially for an accurate polyfit

    # Go through all vids
    test_dir = "assets"
    for filename in os.listdir(test_dir):
        if filename.endswith(".mov"):
            main(f"{test_dir}/{filename}")

    print("\nThanks for watching!\n")