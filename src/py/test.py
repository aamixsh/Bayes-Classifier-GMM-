import numpy as np
from PIL import Image
import os
import random
import math

patchHeight = 64
patchWidth = 64
bins = 8
BoVW_VectorLen = 64
dimension = 0

def calcVectorforPatch(img,xPnt,yPnt):
	vector_ = [0 for i in range(3*bins)]
	for i in range(patchHeight):
		for j in range(patchWidth):
			vector_[int(img[xPnt+i][yPnt+j][0]/32)]+=1;
			vector_[int(8+img[xPnt+i][yPnt+j][1]/32)]+=1;
			vector_[int(16+img[xPnt+i][yPnt+j][2]/32)]+=1;
	return vector_


#create histograms of all the patches and store them into a file in folder output
def createHistograms(folder,filename,img, patchHeight, patchWidth, bins):
	folder="../../data/Output/GMM/Dataset 2/B/featureVectorsBoVW/coast"
	filename = os.path.splitext(filename)[0]
	filename+='.txt'
	z = img.shape
	x=z[0]
	y=z[1]
	xPnt=0
	with open(os.path.join(folder,filename), 'w') as outfile:
		for i in range(int(x/patchHeight)):
			yPnt=0
			for j in range(int(y/patchWidth)):
				vect = calcVectorforPatch(img,xPnt,j*patchWidth)
				outfile.write(' '.join(str(e) for e in vect))
				outfile.write("\n")
			xPnt+=patchHeight
	outfile.close()
		

def calcDist(x,y):
	distance=0
	for i in range(3*bins):
		distance+=(x[i]-y[i])**2
	return math.sqrt(distance)

#It will return the matrix containing mean vectors of all the K-clusters
def K_MeansClustering(filename):
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(3*bins)] for i in range(BoVW_VectorLen)]
	randomKMeans=random.sample(range(0,N-1),BoVW_VectorLen)
	for i in range(BoVW_VectorLen):
		for j in range(3*bins):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	#	Dividing the data of this class to K clusters...
	tempClusters=[[] for i in range(BoVW_VectorLen)]
	totDistance=0
	energy=100
	for i in range(N):
		minDist=np.inf
		minDistInd=0
		for j in range(BoVW_VectorLen):
			Dist=calcDist(tempClass[i],tempClusterMean[j])
			if Dist<minDist:
				minDist=Dist
				minDistInd=j
		tempClusters[minDistInd].append(tempClass[i])
		totDistance+=minDist

	#	Re-evaluating centres until the energy of changes becomes insignificant...
	while energy>60:
		tempClusterMean=[[0 for i in range(3*bins)] for i in range(BoVW_VectorLen)]
		for i in range(BoVW_VectorLen):
			for j in range(len(tempClusters[i])):
				for k in range(3*bins):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(3*bins):
				if len(tempClusters[i])==0:
					#tempClusterMean[i]=
					break;
				else:
					tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(BoVW_VectorLen)]
		newTotDistance=0
		for i in range(N):
			minDist=np.inf
			minDistInd=0
			for j in range(BoVW_VectorLen):
				Dist=calcDist(tempClass[i],tempClusterMean[j])
				if Dist<minDist:
					minDist=Dist
					minDistInd=j
			tempClusters[minDistInd].append(tempClass[i])
			newTotDistance+=minDist
		energy=math.fabs(totDistance-newTotDistance);
		totDistance=newTotDistance;
		print(energy)
	return tempClusterMean


def classify(patchVector,means):
	mini=np.inf
	ind = 0
	for i in range(len(means)):
		dist=calcDist(patchVector,means[i])
		if dist < mini:
			ind = i
			mini = dist
	return ind


#this function will take input a folder and output 2D array having 64 dimensional vector corresponding to each file in folder
def BagofVisualWorlds(folder):
	featureVectors = []
	#concatinating all the files
	outfile=open(os.path.join(folder,"BoVW.txt"), 'w')
	for filename in os.listdir(folder):
		if filename!="BoVW.txt":
			if os.path.splitext(filename)[1] == ".txt":
				infile=open(os.path.join(folder,filename))
				for line in infile:
					outfile.write(line)
				infile.close()
	outfile.close()

	#generating mean vectors from the concatinated file :p
	means = K_MeansClustering(os.path.join(folder,"BoVW.txt"))

	for filename in os.listdir(folder):
		featureVector=[0 for i in range(BoVW_VectorLen)]
		if os.path.splitext(filename)[1] == ".txt":
			infile = open(os.path.join(folder,filename))
			for line in infile:
				number_strings=line.split()
				numbers=[float(n) for n in number_strings]
				clusterNum=classify(numbers,means)
				featureVector[clusterNum]+=1
			infile.close()
			featureVectors.append(featureVector)
	file=open("../../data/Output/GMM/Dataset 2/B/featureVectorsBoVW/coast_BoVW")
	for i in range(len(featureVectors)):
		for j in range(len(featureVectors[i])):
			file.write(str(featureVectors[i][j])+" ")
		file.write("\n")
	return featureVectors


#argument should be a folder name where all the images of a class are present
#functioin should return the matrix containing 64 dimension vector corresponding to each image file in a folder
def generateFeatureVectors(folder):
    # for filename in os.listdir(folder):
    # 	if os.path.splitext(filename)[1] == ".jpg":
    #     	img = Image.open(os.path.join(folder,filename))
	   #      if img is not None:
	   #          createHistograms(folder,filename,np.array(img),patchHeight,patchWidth,bins)
    return BagofVisualWorlds("../../data/Output/GMM/Dataset 2/B/featureVectorsBoVW/coast")


dimension = bins*3

images=generateFeatureVectors("../../data/Input/GMM/Dataset 2/B/train/coast")


print (images)