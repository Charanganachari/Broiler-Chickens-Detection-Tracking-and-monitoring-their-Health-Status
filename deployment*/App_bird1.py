import streamlit as st
import numpy as np
import pandas as pd
import cv2
import time
import datetime
import json
from centroidtracker_pr import CentroidTracker 
import os
import matplotlib.pyplot as plt
import tempfile   
#import mysql.connector
st.set_page_config(page_title='Track & monitor Birds')

st.title('Health Monitoring of Birds')

st.sidebar.subheader("Import Files")
f = st.sidebar.file_uploader(label = "Upload your mp4 file",type = ['mp4'])
tfile = tempfile.NamedTemporaryFile(delete= False)
try:    
    tfile.write(f.read())
except:
    st.write('File readed')    
st.video(tfile.name)

if tfile is not None:
    thermal_camera = cv2.VideoCapture(tfile.name)
    try:
        st.subheader('File uploaded')
    except Exception as e:
        print(e)
        st.write("Please upload file")
    
if st.button('Detect'):
    x_mouse = 0
    y_mouse = 0
    
    def conv_to_temp(pixel_val_1, offset, scale):
        return((pixel_val_1-offset)/scale)

    # set up the thermal camera resolution
    thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
    thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

    # set up the thermal camera to get the gray16 stream and raw data
    thermal_camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y','1','6',' '))
    thermal_camera.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    # set up mouse events and prepare the thermal frame display
    grabbed, frame_thermal = thermal_camera.read()
    cv2.imshow('Temperature_detector', frame_thermal)
    
    net = cv2.dnn.readNet(r"F:\Ai Projects\deployment yolov8\best.onnx")
    file = open("classes.txt","r")
    classes = file.read().split('\n')
    classes1 = ["1WOC","2WOC","3WOC","4WOC"]

    tracker = CentroidTracker(maxDisappeared=200, maxDistance=50)
    tracker1 = CentroidTracker(maxDisappeared=200, maxDistance=50)
    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1
    
    while True:
        # grab the frame from the thermal camera stream
        (grabbed, thermal_frame) = thermal_camera.read()

        
        thermal_frame = cv2.normalize(thermal_frame, thermal_frame, 0, 255, cv2.NORM_MINMAX)
        thermal_frame = np.uint8(thermal_frame)
      
        # colorized the gray8 image using OpenCV colormaps
        thermal_frame = cv2.applyColorMap(thermal_frame, cv2.COLORMAP_INFERNO)
        
        ##################################################################
        #thermal_frame = thermal_camera.read()[1]
        if thermal_frame is None:
            break
        thermal_frame = cv2.resize(thermal_frame, (720,640))
        blob = cv2.dnn.blobFromImage(thermal_frame,scalefactor= 1/255,size=(416,416),mean=[0,0,0],swapRB= True, crop= False)
        net.setInput(blob)
        detections = net.forward()[0]

        frame_count += 1
        total_frames += 1

        # cx,cy , w,h, confidence, 80 class_scores
        # class_ids, confidences, boxes

        classes_ids = []
        confidences = []
        boxes = []
        Data_sheet = {}
        rows = detections.shape[0]

        img_width, img_height = thermal_frame.shape[1], thermal_frame.shape[0]
        x_scale = img_width/416
        y_scale = img_height/416

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.2:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.2:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx- w/2)*x_scale)
                    y1 = int((cy-h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1,y1,width,height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes,confidences,0.2,0.2)
        rects = []

        for i in indices:    
            chicken_box = boxes[i]
            (x1,y1,w,h) = chicken_box.astype("int")
            rects.append(chicken_box)
            label = classes[classes_ids[i]]
            conf = confidences[i]
            text1 = label + "{:.2f}".format(conf)
            age = label
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, w, h = bbox
            x1 = int(x1)
            y1 = int(y1)
            w = int(w)
            h = int(h)

            cv2.rectangle(thermal_frame, (x1, y1), (x1+w,y1+h), (0, 0, 255), 2)
            Bbtemp = thermal_frame [y1+h//2, x1+w//2] 
            temp2 = max(conv_to_temp(Bbtemp,-566.66, 20.238))
            text2 = "{}".format(objectId)
            text3 = "{0:.1f}".format(temp2)
            cv2.putText(thermal_frame, text1, (x1,y1+13),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,255,255),1)
            
            cv2.putText(thermal_frame, text2, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 2)
            time_stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if temp2 >= 40.3:
                cv2.putText(thermal_frame, text3, (x1+100, y1+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1)
                Dgn = "Sick"
            

            else:
                cv2.putText(thermal_frame, text3, (x1+100, y1+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 1)
                Dgn = "Healthy"
            Data_sheet[objectId+1] = {"ID":text2, "Age":label, "Temperature":text3, "Timestamp":time_stamp, "Diagnose":Dgn}
            df = pd.DataFrame.from_dict(Data_sheet, orient = 'index')
                                        
            # write pointer
            cv2.circle(thermal_frame, (x_mouse, y_mouse), 2, (255, 255, 255), -1)
           
            # write temperature
            #cv2.putText(thermal_frame, "{0:.1f} Celsius".format(temp1), (x_mouse - 40, y_mouse - 15), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

        
            if frame_count >= 30:
                end = time.time_ns()
                fps = 10000000000 * frame_count / (end - start)
                frame_count = 0
                start = time.time_ns()
                
        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(thermal_frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Temperature_detector",thermal_frame)
        
        k = cv2.waitKey(10)
        if k == ord('q'):
            break
#    from sqlalchemy import create_engine

#   USER = 'root'
#    PASSWORD = 'PASS123'
#    HOST = 'localhost'
#    PORT = '3306'
#    DATABASE = 'birds'

    # create a SQLAlchemy engine object with the MySQL connection URL
#   engine = create_engine(f'mysql+pymysql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}', echo=True)
#    df.to_sql("detection2", engine, if_exists='append')
#    df.to_excel("Data_sheet.xlsx") 
#    st.write(df)

    # create connection to MySQL database


thermal_camera.release()
cv2.destroyAllWindows()
    