import cv2
import argparse

# parser: to run our code from CMD ==> (pthon yolo2.py --input test.jpg)
# --input: is our argument from CMD
parser = argparse.ArgumentParser()
parser.add_argument('--input', help='path to input image')
args = parser.parse_args()

# min_confidence: minimum confidence threshold
# increase it ==> imporve accuracy, reduce detcetion rate
min_confidence = 0.15
model ='yolov2.weights'
config = 'yolov2.cfg'

# Load our classes from (labels.txt) file and read all classes
classes = None
with open('labels.txt', 'r') as f:
    classes = f.read().rstrip('\n').split('\n')
# print(classes)

# Load weihts and construct graph
net = cv2.dnn.readNetFromDarknet(config, model)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

winName = 'Yolo2 Demo'
cv2.namedWindow(winName), cv2.WINDOW_NORMAL

# Read Image from --input arg in CMD
myFrame = cv2.imread(args.input)

# Get width and Height
height, width, ch = myFrame.shape

# Create a 4D blob from a Frame
blob = cv2.dnn.blobFromImage(myFrame, 1.0/255.0, (416,416) , True, crop=False)
net.setInput(blob)

# Run the preprocessed input input blog through the network
predictions = net.forward()
probability_index = 5


# shape[0]:    the objects that detected
# class_index: the index of each item in our classes file
for i in range(predictions.shape[0]):
    probability_arrange = predictions[i][probability_index:]
    class_index         = probability_arrange.argmax(axis=0)
    confidence          = probability_arrange[class_index]

    # confirm that the object detected 100%
    if confidence > min_confidence:   
        x_center  = predictions[i][0] * width
        y_center  = predictions[i][1] * height
        width_box = predictions[i][2] * width
        height_box= predictions[i][3] * height

        x1= int(x_center - width_box * 0.5)
        y1= int(y_center - height_box * 0.5)
        x2= int(x_center + width_box * 0.5)
        y2= int(y_center + height_box * 0.5)

        # Drow RED rectangle around the object detected
        cv2.rectangle(myFrame, (x1,y1), (x2,y2), (255, 255, 255), 1)

        # Write object name above the rectangle
        cv2.putText(myFrame,
                    classes[class_index] + " " + "{0:.1f}".format(confidence),  # To which class this objec belong
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,   # Text Font
                    1,
                    (255, 255, 255),            # Text Color
                    1,
                    cv2.LINE_AA)

cv2.imshow(winName, myFrame)
if(cv2.waitKey() >= 0):
    cv2.destroyAllWindows()