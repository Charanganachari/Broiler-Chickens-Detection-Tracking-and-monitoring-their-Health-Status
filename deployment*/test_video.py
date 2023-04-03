import cv2
import numpy as np

cap = cv2.VideoCapture(r"E:\AI_project\Thermalvideo1.mp4")
net = cv2.dnn.readNetFromONNX(r"E:\AI_project\bestmulti\AI_woc_woa_best\runs\train\exp\weights\best.onnx")
classes = ["1WOC","2WOC","3WOC","4WOC"]
print(classes)
img = cap.read()[1]

img = cv2.resize(img, (416,416))
blob = cv2.dnn.blobFromImage(img,size=(416,416),scalefactor= 1/255,mean=[0,0,0],swapRB= True, crop= False)
net.setInput(blob)
detections = net.forward()[0]
while True:
    img = cap.read()[1]
    if img is None:
        break
    img = cv2.resize(img, (416,416))
    blob = cv2.dnn.blobFromImage(img,size=(416,416),scalefactor= 1/255,mean=[0,0,0,],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
  

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/416
    y_scale = img_height/416

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if (confidence >= 0.2).all():
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

    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,0),2)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.5,(255,0,255),1)

    cv2.imshow("VIDEO",img)
    k = cv2.waitKey(10)
    if k == ord('q'):
        break