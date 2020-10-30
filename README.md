# Yolo2

Yolo is one of the most sucessful object detection algorithm in the field, known for its lightening speed and decent accuracy. Comparing to other regional proposal frameworks that detect objects region by region, which requires many times of feature extraction, the input images are processed once in Yolo.

Yolo2 uses a VGG-style CNN called the DarkNet as feature extractors. Please note that DarkNet is an umbrella of various networks, and people use different variants to increase speed or accuracy.

source: https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088

tutorial: https://www.youtube.com/watch?v=w0tDDFip7KM

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


# To Run this code:

1. Download the yolov2.weights, labels.txt, yolov2.cfg, text.jpg files: 

2. Now run using the following command:

  python3 yolo2.py --input test.jpg

