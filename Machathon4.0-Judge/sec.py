"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import numpy as np
import keyboard
import math
import logging

from machathon_judge import Simulator, Judge


class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def stabilize_steering_angle(
          curr_steering_angle, 
          new_steering_angle, 
          num_of_lane_lines, 
          max_angle_deviation_two_lines=5, 
          max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    if new angle is too different from current angle, 
    only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2 :
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else :
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane
    
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering_angle = new_steering_angle
    return stabilized_steering_angle

def warpImg(img, points, w, h, inv=False):
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2, pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    return imgWarp

def nothing(a):
    pass

def initializeTrackbars(initialTrackbarVals, wT=640, hT=480):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initialTrackbarVals[0], wT // 2, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initialTrackbarVals[1], hT, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initialTrackbarVals[2], wT // 2, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initialTrackbarVals[3], hT, nothing)

def valTrackbars(wT=640, hT=480):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT - widthTop, heightTop),
                         (widthBottom, heightBottom), (wT - widthBottom, heightBottom)])
    return points

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape

    # figure out the heading line from steering angle
    # heading line (x1,y1) is always center bottom of the screen
    # (x2, y2) requires a bit of trigonometry

    # Note: the steering angle of:
    # 0-89 degree: turn left
    # 90 degree: going straight
    # 91-180 degree: turn right 
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)

    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

    return heading_image

def get_edges(components):

    first_sobelx = cv2.Sobel(components[0].astype("uint8"), ddepth=cv2.CV_64F, dx=1, dy=0,ksize=5)
    first_sobely = cv2.Sobel(components[0].astype("uint8"), ddepth=cv2.CV_64F, dx=0, dy=1,ksize=5)

    second_sobelx = cv2.Sobel(components[1].astype("uint8"), ddepth=cv2.CV_64F, dx=1, dy=0,ksize=5)
    second_sobely = cv2.Sobel(components[1].astype("uint8"), ddepth=cv2.CV_64F, dx=0, dy=1,ksize=5)
    return (first_sobelx+first_sobely),(second_sobelx+second_sobely)
def fit_poly(edges, degree):
    points0 =  edges[0].nonzero()
    fit0 = np.polyfit(points0[0],points0[1], degree)

    points1 =  edges[1].nonzero()
    fit1 = np.polyfit(points1[0],points1[1], degree)

    return fit0,fit1

def get_lane_mask(image):
    b, g, r = cv2.split(image)
    _, road_mask_b = cv2.threshold(b, 220, 255, cv2.THRESH_BINARY)
    _, road_mask_g = cv2.threshold(g, 220, 255, cv2.THRESH_BINARY)
    _, road_mask_r = cv2.threshold(r, 220, 255, cv2.THRESH_BINARY)
    road_mask = cv2.bitwise_and(road_mask_b, road_mask_g)
    road_mask = cv2.bitwise_and(road_mask, road_mask_r)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((80,80),np.uint8))

    _, lane_mask_b = cv2.threshold(b, 70, 255, cv2.THRESH_BINARY_INV)
    _, lane_mask_g = cv2.threshold(g, 70, 255, cv2.THRESH_BINARY_INV)
    _, lane_mask_r = cv2.threshold(r, 140, 255, cv2.THRESH_BINARY)
    lane_mask = cv2.bitwise_and(lane_mask_b, lane_mask_g)
    lane_mask = cv2.bitwise_and(lane_mask, lane_mask_r)
    lane_mask = cv2.bitwise_and(lane_mask, road_mask)
    cv2.imshow("lane_mask",lane_mask)
    return lane_mask
def get_steering_angle(image):
    lane_mask = get_lane_mask(image)
    output = cv2.connectedComponentsWithStats(
    lane_mask, 8, cv2.CV_32S)
    num_components,labels, stats, centroids = output

    num_pixels = np.array([np.sum(labels == i) for i in range(1,num_components) ])
    labels_of_bigest_2components = num_pixels.argsort()[::-1][:2]+1
    components = [ (labels == id) for id in labels_of_bigest_2components]
    cv2.imshow("components",(components[0]+components[1]).astype("uint8") * 255)

    first_poly,second_poly = fit_poly(get_edges(components),2)
    average_poly = (first_poly+second_poly)/2

    derivative = np.polyder(average_poly)
    first_slope = np.polyder(first_poly)
    second_slope = np.polyder(second_poly)

    slope = np.polyval(derivative,340)
    return first_poly,second_poly,average_poly,first_slope,second_slope,slope

def drawPoints(image,points):
    for point in points:
        cv2.circle(image, (int(point[0]),int(point[1])), 5, (0,0,255), -1)
    return image

def run_car(simulator: Simulator) -> None:
    """
    Function to control the car using keyboard

    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    fps_counter.step()
    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    out_img = img.copy()
    try:
        first_poly,second_poly,average_poly,first_slope,second_slope,slope = get_steering_angle(out_img)
        steering_angle = np.arctan(slope)*180/np.pi
        heading_img = display_heading_line(out_img,steering_angle)
        cv2.imshow("heading_img",heading_img)
        lspace = np.linspace(0,  out_img.shape[0]-1, out_img.shape[0])
        out_img = cv2.polylines(out_img, [np.array([np.polyval(first_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0))
        out_img = cv2.polylines(out_img, [np.array([np.polyval(second_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0))
        out_img = cv2.polylines(out_img, [np.array([np.polyval(average_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0))
        cv2.imshow("out_img",out_img)
    except:
        #steering_angle=np.arctan()
        print("error")
    lane_mask = get_lane_mask(img)
    h,w,_ = img.shape
    points=valTrackbars()
    imgWarp = warpImg(img,points,w,h)
    laneWarp = warpImg(lane_mask,points,w,h)
    imgWarpPoints = drawPoints(img,points)
    cv2.imshow("imgWarp",imgWarp)
    cv2.imshow("imgWarpPoints",imgWarpPoints)
    cv2.imshow("laneWarp",laneWarp)

    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("image", img)
    #cv2.imshow("Lanes", lanes_img)
    cv2.waitKey(1)
    
    # Control the car using keyboard
    steering = 0
    if keyboard.is_pressed("a") or keyboard.is_pressed("left"):
        steering = 1
    elif keyboard.is_pressed("d") or keyboard.is_pressed("right"):
        steering = -1
    
    
    
    throttle = 0
    if keyboard.is_pressed("w") or keyboard.is_pressed("up"):
        throttle = 0.3
    elif keyboard.is_pressed("s") or keyboard.is_pressed("down"):
        throttle = -0.3
    
    #steering=mapped_steering_angle
    #set_point=5*(1-abs(steering))
    #if set_point<1:
    #set_point=1
    #error_speed=set_point-simulator.get_state()[1]
    #throttle=error_speed*0.1
    #print(steering,set_point,simulator.get_state()[1])
    simulator.set_car_steering(steering * simulator.max_steer_angle / 1.7)
    simulator.set_car_velocity(throttle * 25)

    #print(simulator.get_state())



if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()
    initialTrackBarVals=[211,314,0,378]
    initializeTrackbars(initialTrackBarVals)
    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="your_new_team_code", zip_file_path="your_solution.zip")
    first_frame = True
    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
