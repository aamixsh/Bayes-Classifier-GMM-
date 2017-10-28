#	CS669 - Assignment 2 (Group-2) 
#	Last edit: 28/10/17
#	About: 
#		This program extracts 2-D feature vectors from B/W training images for a given overlapping patch size.

import numpy as np
import math
import os
import random
from PIL import Image

patchSize=7									#	Size of overlapping patches in the images.

#	Extracts features from images.
def calcPrereq(direct,directOut):

	#	Reading images...
	tempData=[]
	for filename in os.listdir(direct):
		image=Image.open(direct+filename)
		if image is not None:
			tempData.append(np.array(image))
	N=len(tempData)

	#	Making patches and writing feature vectors to a file...
	file=open(os.path.join(directOut,"trainingFeatures.txt"),"w")
	for n in range(N):
		imgSize=np.shape(np.array(tempData[n]))
		start_x=0
		for end_x in range(patchSize-1,imgSize[0]):
			
			#	First calculating data for first patch.
			start_y,end_y=0,patchSize-1
			tempMean=0.0
			tempVariance=0.0
			
			for i in range(start_x,end_x+1):
				for j in range(start_y,end_y+1):
					tempMean+=tempData[n][i][j]
			tempMean/=patchSize**2
			
			for i in range(start_x,end_x+1):
				for j in range(start_y,end_y+1):
					tempVariance+=(tempData[n][i][j]-tempMean)**2
			tempVariance/=patchSize**2
			
			file.write(str(tempMean)+" "+str(tempVariance)+"\n")

			#	Now using it to calculate data for next patches in that row, efficiently.
			for end_y in range(patchSize,imgSize[1]):
				oldMean=tempMean
				oldVariance=tempVariance
				newMean=oldMean*patchSize**2
				
				for x in range(start_x,end_x+1):
					newMean-=tempData[n][x][start_y]
				for x in range(start_x,end_x+1):
					newMean+=tempData[n][x][end_y]
				
				newMean/=patchSize**2
				newVariance=oldVariance*patchSize**2
				newVariance+=patchSize**2*(newMean**2-oldMean**2)
				newVariance-=2*(patchSize**2)*oldMean*(newMean-oldMean)
				
				for x in range(start_x,end_x+1):
					newVariance+=(tempData[n][x][end_y]-newMean)**2
				for x in range(start_x,end_x+1):
					newVariance-=(tempData[n][x][start_y]-newMean)**2
				
				newVariance/=patchSize**2
				start_y=start_y+1
				file.write(str(tempMean)+" "+str(tempVariance)+"\n")
				tempMean=newMean
				tempVariance=newVariance
			start_x=start_x+1
	file.close()

#	Program starts here...
print ("\nThis program extracts 2-D feature vectors from B/W training images for a given overlapping patch size.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for training input and output or default (o/d): ")

direct=""
directOtrain=""

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training images directory: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store these feature vectors: ")
	directOtrain=os.path.join(directO,"train")
else:
	direct="../../data/Input/Clustering/Dataset 2/C/train"
	directOtrain="../../data/Output/Clustering/Dataset 2/C/featureVectorsTrain"

patchSize=input("Enter the size of a single patch [square] in pixels: ")

if direct[len(direct)-1]!='/':
	direct+="/"
if directOtrain[len(directOtrain)-1]!='/':
	directOtrain+="/"

print "Reading images and generating feature vectors from training data (this may take a while)..."
calcPrereq(direct,directOtrain)

print "Features extracted successfully."

#	End.