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

def get_edges(component):

    sobelx = cv2.Sobel(component.astype("uint8"), ddepth=cv2.CV_64F, dx=1, dy=0,ksize=5)
    sobely = cv2.Sobel(component.astype("uint8"), ddepth=cv2.CV_64F, dx=0, dy=1,ksize=5)
    return (sobelx+sobely)

def fit_poly(edge, degree):
    points0 =  edge.nonzero()
    fit = np.polyfit(points0[0],points0[1], degree)
    return fit

def get_slope(image,degree = 2):
    global last_output
    b, g, r = cv2.split(image)
    #r1,r2,g1,g2,b1,b2=valTrackbars2()
    _, road_mask_b = cv2.threshold(b, 200, 255, cv2.THRESH_BINARY)
    _, road_mask_g = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    _, road_mask_r = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)
    road_mask = cv2.bitwise_and(road_mask_b, road_mask_g)
    road_mask = cv2.bitwise_and(road_mask, road_mask_r)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, np.ones((10,10),np.uint8))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((80,80),np.uint8))
    #cv2.imshow("road_mask2",road_mask)

    _, lane_mask_b = cv2.threshold(b, 70, 255, cv2.THRESH_BINARY_INV)
    _, lane_mask_g = cv2.threshold(g, 70, 255, cv2.THRESH_BINARY_INV)
    _, lane_mask_r = cv2.threshold(r, 140, 255, cv2.THRESH_BINARY)
    lane_mask = cv2.bitwise_and(lane_mask_b, lane_mask_g)
    lane_mask = cv2.bitwise_and(lane_mask, lane_mask_r)
    lane_mask = cv2.bitwise_and(lane_mask, road_mask)

    #cv2.imshow("lane_mask",lane_mask)
    #road_mask=get_road(image)
    output = cv2.connectedComponentsWithStats(
    lane_mask, 8, cv2.CV_32S)
    num_components,labels, stats, centroids = output
    num_pixels = np.array([np.sum(labels == i) for i in range(1,num_components) ])
    if num_components < 2:
        return last_output
    if num_components > 2:
      if np.all(np.sort(num_pixels)[::-1][:2]>1000):
        labels_of_bigest_2components = num_pixels.argsort()[::-1][:2]+1
        components = [ (labels == id) for id in labels_of_bigest_2components]

        #cv2.imshow("component",components[0].astype("uint8")*255+components[1].astype("uint8")*255)

        first_edge = get_edges(components[0])
        second_edge = get_edges(components[1])

        #cv2.imshow("edge",first_edge+second_edge)

        first_poly = fit_poly(first_edge,degree)
        second_poly = fit_poly(second_edge,degree)

        average_poly = (first_poly+second_poly)/2

        #last point at which fitted polynomials are accurate use it as a lookhead 
        y0 = min(np.max(first_edge.nonzero()[0]),np.max(second_edge.nonzero()[0]))-10
        x0 = np.polyval(average_poly,y0)
        #Lower mid point (represents car)
        y1 = image.shape[0]
        x1 = int(image.shape[1]/2)
        #print(image.shape[0],image.shape[1])
        #drawPoints()
        slope = (y1-y0)/(x1-x0)
        steering_angle= math.atan2(y1-y0,x1-x0)
        return first_poly,second_poly,slope,[x0,y0],steering_angle,2
    
    labels_of_bigest_component = num_pixels.argsort()[::-1][:1]+1
    component = labels == labels_of_bigest_component

    #cv2.imshow("component",component.astype("uint8")*255)
    edge = get_edges(component)
    #cv2.imshow("edge",edge)
    poly = fit_poly(edge,degree)
    #last point at which fitted polynomials are accurate use it as a lookhead 
    y0 = np.max(edge.nonzero()[0])-90
    x0 = np.polyval(poly,y0)

    y1 = y0+30
    x1 = np.polyval(poly,y1)
    slope = (y1-y0)/(x1-x0)
    steering_angle= math.atan2(y1-y0,x1-x0)
    #dervitive = np.polyder(np.poly1d(poly))
    #slope = dervitive(y0)
    return poly,poly,slope,[x0,y0],steering_angle,1

last_output = None
last_steering_angle = 0
#last_time=time.monotonic()
#time_now=time.monotonic()
#dt_list=[]
#is_curve=False
dsteering_list=[]

def run_car(simulator: Simulator) -> None:
    global last_output,last_steering_angle,dsteering_list#,last_time,dt_list,is_curve,time_now
    #if is_curve:
    #    min_dt=100
    #else:
    #    min_dt=10
    #dt=0
    #while dt<min_dt:
    #time_now = time.monotonic()
    #dt=(time_now-last_time)*1000
    #dt_list.append(dt)
    #print("dtime:",dt,"ms","is_curve:",is_curve)
    #last_time=time_now
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
    #try:
    #fps_counter.step()
    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #out_img = img.copy()
    
    last_output = get_slope(img)
    first_poly,second_poly,slope,point,steering_angle,number_lanes = last_output
    
    #average_poly = (first_poly+second_poly)/2
    #lspace = np.linspace(0,  out_img.shape[0]-1, out_img.shape[0])
    #out_img = cv2.polylines(out_img, [np.array([np.polyval(first_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0),thickness=3)
    #out_img = cv2.polylines(out_img, [np.array([np.polyval(second_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0),thickness=3)
    #out_img = cv2.polylines(out_img, [np.array([np.polyval(average_poly,lspace),lspace],dtype=np.int32).T], False, (0,0,0),thickness=3)
    #out_img = cv2.circle(out_img.copy(), (int(point[0]),point[1]), radius=5, color=(0, 0, 255), thickness=-1)
    #out_img = cv2.line(out_img.copy(), (out_img.shape[1],int(point[1])), (0,int(point[1])), (0, 0, 255), thickness=3)
    #out_img = display_heading_line(out_img,steering_angle)
    
    
    steering = (0.5*np.pi-steering_angle)/4.5
    #print("steering_angle",steering_angle*180/np.pi,"steering",steering*180/np.pi)
    dsteering = abs(steering-last_steering_angle)/(np.pi/4.5)
    dsteering_list.append(dsteering)
    dsteering_mean = np.mean(dsteering_list[-10:])
    #print("dsteering",dsteering,"dsteering_mean",dsteering_mean)
    factor=4*(steering/(np.pi*0.5/4.5))**2#1-math.log10(2*dsteering+1.4)
    new_steering = factor*steering
    
    sign = np.sign(new_steering)
    if abs(new_steering)>np.pi*0.5/5.5:
        new_steering = sign*np.pi*0.5/7
    """
    if abs(new_steering)<np.pi*0.5/12:
        new_steering = steering*0.1
    elif abs(new_steering)<np.pi*0.5/6:
        new_steering = steering*0.2
    """
    #print("last_steering_angle",last_steering_angle*180/np.pi,"steering",steering*180/np.pi,"dsteering",dsteering,"factor",factor,"new_steering",new_steering*180/np.pi)
    
    #stable_steering_angle = steering_angle#stabilize_steering_angle(last_steering_angle, steering_angle, number_lanes,20,20)
    #out_img = display_heading_line(out_img,0.5*np.pi-steering,line_color=(255,0,255))
    #out_img = display_heading_line(out_img,0.5*np.pi-last_steering_angle,line_color=(255,0,0))
    #out_img = display_heading_line(out_img,0.5*np.pi-new_steering,line_color=(0,255,0))
    #print("steering_angle",steering_angle,"stable_steering_angle",stable_steering_angle,"last_steering_angle",last_steering_angle)
    last_steering_angle = new_steering
    
    #cv2.imshow("out_img",out_img)
    
    #lane_mask = get_lane_mask(img.copy())
    #h,w,_ = img.shape
    #points=valTrackbars()
    #imgWarp = warpImg(img,points,w,h)
    #laneWarp = warpImg(lane_mask,points,w,h)
    #road_mask=get_road(img.copy())
    #imgWarpPoints = drawPoints(img,points)
    #cv2.imshow("imgWarp",imgWarp)
    #cv2.imshow("imgWarpPoints",imgWarpPoints)
    #cv2.imshow("laneWarp",laneWarp)
    
    #cv2.imshow("road_mask",road_mask)

    #fps = fps_counter.get_fps()
    #print("fps: ",fps)
    # draw fps on image
    #cv2.putText(
    #    out_img,
    #    f"FPS: {fps:.2f}",
    #    (10, 30),
    #    cv2.FONT_HERSHEY_SIMPLEX,
    #    1,
    #    (0, 255, 0),
    #    2,
    #    cv2.LINE_AA,
    #)

    
    sinc = 1.8*math.sin(math.sin(2*math.pi*new_steering)/(2*math.pi*new_steering))**2
    throttle = 7*sinc-1
    #print("steering",steering*180/np.pi,"throttle",throttle,"slope",slope)
    if throttle>7:
        throttle*=1.5
        new_steering*=0.86
    if abs(new_steering)<np.pi*0.5/9:
        new_steering = new_steering*0.19

    
    #if abs(steering)>np.pi*0.5/7:
    #    is_curve = True
    #else:
    #    is_curve = False
    #throttle*=3
    #if is_curve:
    #    throttle = 0.1*throttle
    jump=False
    if dsteering>(dsteering_mean+0.14):
        #print("jump")
        throttle=-50
        #new_steering*=0.2
        jump=True
        #new_steering *= 0
        #dsteering = dsteering_mean+0.15
    throttle*=1.13
    if jump==True:
        count=200
    else :
        count=1
    while count>0:
        simulator.set_car_steering(new_steering)
        simulator.set_car_velocity(throttle)#*(dt/60))
    #if jump==True:
    #    time.sleep(0.099)
        count-=1
    #cv2.imshow("image", out_img)
    #cv2.waitKey(1)


if __name__ == "__main__":
    # Initialize any variables needed
    #cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()
#    initialTrackBarVals=[211,314,0,378]
    #initializeTrackbars(initialTrackBarVals)
#    initialTrackBarVals2=[130,170,130,170,130,170]
    #initializeTrackbars2(initialTrackBarVals2)
    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="XXHFOprCe", zip_file_path="L:\Roro\college\senior2\STPMacathon40\simulation\Machathon4.0-Judge\submission2.zip")
    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
    #print(np.mean(dt_list),np.std(dt_list))
