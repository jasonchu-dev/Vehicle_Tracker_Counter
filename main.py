import cv2 as cv
import numpy as np
from scipy import spatial

detect = []
FRAMES_BEFORE_CURRENT = 100


def center(x, y, w, h):
    x1 = int(w//2)
    y1 = int(h//2)
    cx = x + x1
    cy = y + y1
    return cx, cy


def was_in_previous(previous_frame_detections, c, w, h, current_detections):
    distance = np.inf
    for i in range(FRAMES_BEFORE_CURRENT):
        coords = list(previous_frame_detections[i].keys())
        if len(coords) == 0: continue
        temp_dist, index = spatial.KDTree(coords).query([(c[0], c[1])])
        if temp_dist < distance:
            distance = temp_dist
            frame_num = i
            coord = coords[index[0]]
        
    if distance > max(w, h) / 2: return False
    current_detections[(c[0], c[1])] = previous_frame_detections[frame_num][coord]
    return True


def vehicle_count(bbox, indices, count, previous_frame_detections, classes):
    current_detections = {}
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1 = bbox[i][0], bbox[i][1]
            w1, h1 = bbox[i][2], bbox[i][3]
            c = center(x1, y1, w1, h1)
            # horizontal reference line
            # if (classes[i] == 'car' or classes[i] == 'truck' or classes[i] == 'bus' or classes[i] == 'motorcycle') and c1[1] < 400 + 100 and c1[1] > 400 - 100:
            # if (classes[i] == 'car' or classes[i] == 'truck' or classes[i] == 'bus' or classes[i] == 'motorcycle') and c1[0] < 400 + 100 and c1[1] < 400 + 100:
            # vertical reference line
            if (classes[i] == 'car' or classes[i] == 'truck' or classes[i] == 'bus' or classes[i] == 'motorcycle') and c[0] < 400 + 100 and c[0] > 400 - 100:
            # if (classes[i] == 'car' or classes[i] == 'truck' or classes[i] == 'bus' or classes[i] == 'motorcycle'):
                current_detections[(c[0], c[1])] = count
                if not was_in_previous(previous_frame_detections, c, w1, h1, current_detections):
                    count += 1

                id = current_detections.get((c[0], c[1]))

                if (list(current_detections.values()).count(id) > 1):
                    current_detections[(c[0], c[1])] = count
                    count += 1

    return count, current_detections



def find(outputs, frame, classes, previous_frame_detections, count):
    global detect
    ht, wt, ct = frame.shape
    bbox = []
    ids = []
    confidence = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            id = np.argmax(scores)
            conf = scores[id]
            if conf > 0:
                w, h = int(detection[2]*wt), int(detection[3]*ht)
                x, y = int((detection[0]*wt)-w/2), int((detection[1]*ht)-h/2)
                bbox.append([x, y, w, h])
                ids.append(id)
                confidence.append(float(conf))

    indices = cv.dnn.NMSBoxes(bbox, confidence, 0.5, 0.3)

    if len(indices) > 0:
        for i in indices.flatten():
			# extract the bounding box coordinates
            x, y, w, h = bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.putText(frame, f'{classes[ids[i]].upper()} {int(confidence[i]*100)}%', (x, y-10), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (255, 0, 0), 2)


    count, current_detections = vehicle_count(bbox, indices, count, previous_frame_detections, classes)
    cv.putText(frame, "Car Count: "+str(count), (100, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

    return count, current_detections


def main():
    count = 0
    global FRAMES_BEFORE_CURRENT

    classesFile = 'C:/Users/Whata/source/repos/CalTrans_Research/Object_Tracking/coco.names'
    classes = []
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNetFromDarknet('C:/Users/Whata/source/repos/CalTrans_Research/Object_Tracking/yolov3.cfg', 'C:/Users/Whata/source/repos/CalTrans_Research/Object_Tracking/yolov3.weights')
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    vid = cv.VideoCapture("C:/Users/Whata/source/repos/CalTrans_Research/Object_Tracking/vid.MOV")

    previous_frame_detections = [{(0,0):0} for i in range(FRAMES_BEFORE_CURRENT)]

    frame_width = int(vid.get(3))
    frame_height = int(vid.get(4))
    size = (frame_width, frame_height)
   
    # Below VideoWriter object will create
    # a frame of above defined The output 
    # is stored in 'filename.avi' file.
    result = cv.VideoWriter('C:/Users/Whata/source/repos/CalTrans_Research/Object_Tracking/output/output1.avi', 
                         cv.VideoWriter_fourcc(*'XVID'),
                         60, size)

    while 1:
        ret, frame = vid.read()
        if not ret: break
        cv.rectangle(frame, (0, 300), (1280, 500), (0, 0, 255), 2)
        cv.line(frame, (0, 400), (1280, 400), (0, 0, 255), 2)
        # print(frame.shape)
        blob = cv.dnn.blobFromImage(frame, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        count, detections = find(outputs, frame, classes, previous_frame_detections, count)
        cv.putText(frame, "Car Count: "+str(count), (100, 80), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
        
        result.write(frame)

        cv.imshow("Frame", frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

        previous_frame_detections.pop(0)
        previous_frame_detections.append(detections)

    vid.release()
    result.release()
    cv.destroyAllWindows()

    print(count)

if __name__ == "__main__": 
    main()