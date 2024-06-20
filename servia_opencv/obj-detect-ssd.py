import cv2
thres = 0.5  # Threshold to detect object
#cap = cv2.VideoCapture('../class296a-2/resources/soccer.mp4') #if you use a file
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #3
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) #4
with open('coco.names','rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    configFile = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsFile = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsFile, configFile)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    while True:
        ret, img = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        #print(classIds, bbox)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(255, 255, 0), thickness=8)
                cv2.putText(img, classNames[classId-1].upper(),(box[0]+2,box[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv2.putText(img, str(round(confidence * 100, 2)), (box [0]+2,box[1]+22), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 255), 1)
        cv2.imshow("Overlay", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
