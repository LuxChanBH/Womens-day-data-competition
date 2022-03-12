# python recognize.py  --image path/to/image.jpg

# import libraries
from operator import index
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import pandas as pd

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="input image path")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("Loading Face Detector...")
protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("Loading Face Recognizer...")
embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
le = pickle.loads(open('output/le.pickle', "rb").read())

# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions	
print("Hellooooo"*10)
prediction={'ID':[],'PRED':[]}
for files in os.listdir(args["image"]):
	image = cv2.imread(args["image"]+files)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	confidence=0.5
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
		confidence2 = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence2 > confidence:
			confidence=confidence2
			# compute the (x, y)-coordinates of the bounding box for the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob through our face embedding model to obtain the 128-d quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output image
	prediction['ID']+=[files]
	prediction['PRED']+=[name]
	cv2.imshow("Image", image)
	cv2.waitKey(10)
	cv2.imwrite(os.getcwd()+"/Predict_Out/"+files,image)
print(prediction)

df = pd.DataFrame.from_dict(prediction)
df.columns=['ID','PRED']

def stripunit(tdict):
    for i in tdict:
        df[i]= df[i].apply(lambda x : int(x.strip(tdict[i]))) 
 
D = {'ID':'.jpg' } 


stripunit(D)

df=df.sort_values(by=["ID"])
print("Im giving head")
print(df.head())
print(df.info())
print(os.getcwd()+"/Submissions.csv")
df.to_csv(os.getcwd()+"/Submissions.csv",index=False)
