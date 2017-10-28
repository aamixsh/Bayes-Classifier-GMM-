#	CS669 - Assignment 2 (Group-2) 
#	Last edit: 28/10/17
#	About: 
#		This program extracts color histogram feature vectors from images for a given non-overlapping patch size.

import numpy as np
import math
import os
import random
from PIL import Image

patchHeight=64					#	Height of patches in pixels to be extracted from images.
patchWidth=64					#	Height of patches in pixels to be extracted from images.
bins=8							#	Number of bins of each color to decide the dimension of feature vectors to be extracted.
dimension=0						#	dimension of feature vectors.

#	Returns feature vector from a given patch of image.
def calcVectorforPatch(img,xPnt,yPnt):
	Vector = [0 for i in range(dimension)]
	for i in range(patchHeight):
		for j in range(patchWidth):
			Vector[int(img[xPnt+i][yPnt+j][0]/32)]+=1
			Vector[int(8+img[xPnt+i][yPnt+j][1]/32)]+=1
			Vector[int(16+img[xPnt+i][yPnt+j][2]/32)]+=1
	return Vector

#	Stores all feature vectors of an image in corresponding file.
def createHistograms(folder,filename,img):
	filename=os.path.splitext(filename)[0]
	filename+='.txt'
	z=img.shape
	x,y=z[0],z[1]
	xPnt=0
	outfile=open(os.path.join(folder,filename),"w")
	for i in range(int(x/patchHeight)):
		yPnt=0
		for j in range(int(y/patchWidth)):
			vect = calcVectorforPatch(img,xPnt,j*patchWidth)
			outfile.write(' '.join(str(e) for e in vect))
			outfile.write("\n")
		xPnt+=patchHeight
	outfile.close()

def extractFeature(inpDir,outDir):
	for contents in os.listdir(inpDir):
		contentName=os.path.join(inpDir,contents)
		if os.path.isdir(contentName):
			print "Inside "+contents
			for filename in os.listdir(contentName):
				img=Image.open(os.path.join(contentName,filename))
				if img is not None:
					print "Read "+filename+"."
					outputFilename=os.path.join(outDir,contents,filename)
					if not os.path.exists(os.path.dirname(outputFilename)):
						try:
							os.makedirs(os.path.dirname(outputFilename))
						except OSError as exc:
							if exc.errorno!=errorno.EEXIST:
								raise
					print "Extracting features..."
					createHistograms(os.path.join(outDir,contents),filename,np.array(img))
					print "Done."

#	Program starts here...
print ("\nThis program extracts color histogram feature vectors from images for a given non-overlapping patch size.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for training input and output or default (o/d): ")

direct=""
directT=""
directOtrain=""
directOtest=""

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training images directory for all classes: ")
	directT=raw_input("Enter the path (relative or complete) of the test images directory for all classes: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store these feature vectors: ")
	directOtrain=os.path.join(directO,"train")
	directOtest=os.path.join(directO,"test")
else:
	direct="../../data/Input/GMM/Dataset 2/B/train"
	directT="../../data/Input/GMM/Dataset 2/B/test"
	directOtrain="../../data/Output/GMM/Dataset 2/B/featureVectorsCH/train"
	directOtest="../../data/Output/GMM/Dataset 2/B/featureVectorsCH/test"

patchHeight=input("Enter the height of a single patch in pixels: ")
patchWidth=input("Enter the height of a single patch in pixels: ")
bins=input("Enter the number of bins to be used to generate color-histogram of each patch: ")
dimension=3*bins

if direct[len(direct)-1]!='/':
	direct+="/"
if directT[len(directT)-1]!='/':
	directT+="/"
if directOtrain[len(directOtrain)-1]!='/':
	directOtrain+="/"
if directOtest[len(directOtest)-1]!='/':
	directOtest+="/"

print "Generating feature vectors of training images..."
extractFeature(direct,directOtrain)

print "Now generating feature vectors of test images..."
extractFeature(directT,directOtest)

print "Read everything successfully."

#	End.