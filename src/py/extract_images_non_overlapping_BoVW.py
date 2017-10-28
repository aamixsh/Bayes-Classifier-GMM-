#	CS669 - Assignment 2 (Group-2) 
#	Last edit: 28/10/17
#	About: 
#		This program extracts Bag-of-Visual-Words feature vectors of images from their color histogram features.

import numpy as np
import math
import os
import random
from PIL import Image

patchHeight=64					#	Height of patches in pixels to be extracted from images.
patchWidth=64					#	Height of patches in pixels to be extracted from images.
bins=8							#	Number of bins of each color to decide the dimension of feature vectors to be extracted.
BoVW_VectorLen=64				#	Number of clusters to divide all feature vectors of images to get the dimension.
dimension=24					#	dimension of feature vectors.
	
#	Distance between two points in 'dimension' dimensional space.
def dist(x,y):
	distance=0
	for i in range(dimension):
		distance+=(x[i]-y[i])**2
	return math.sqrt(distance)

#	Returns mean vectors of all the K-clusters.
def kMeansClustering(filename):
	K=BoVW_VectorLen
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)
	file.close()

	#	K-means clustering for initiating GMM formation...

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	randomKMeans=random.sample(range(0,N-1),K)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	#	Dividing the data of this class to K clusters...
	tempClusters=[[] for i in range(K)]
	totDistance=0
	energy=np.inf
	for i in range(N):
		minDist=np.inf
		minDistInd=0
		for j in range(K):
			Dist=dist(tempClass[i],tempClusterMean[j])
			if Dist<minDist:
				minDist=Dist
				minDistInd=j
		tempClusters[minDistInd].append(tempClass[i])
		totDistance+=minDist
	
	#	Re-evaluating centres until the energy of changes becomes insignificant (convergence)...
	while energy>1000:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				tempClusterMean[i][k]/=len(tempClusters[i])
		tempClusters=[[] for i in range(K)]
		newTotDistance=0
		for i in range(N):
			minDist=np.inf
			minDistInd=0
			for j in range(K):
				Dist=dist(tempClass[i],tempClusterMean[j])
				if Dist<minDist:
					minDist=Dist
					minDistInd=j
			tempClusters[minDistInd].append(tempClass[i])
			newTotDistance+=minDist
		energy=math.fabs(totDistance-newTotDistance)
		totDistance=newTotDistance
	return tempClusterMean

#	Returns the index of the cluster in which the vector 'CHVector' is classified.
def classify(CHVector,clusters):
	minInd=np.inf
	ind=0
	for i in range(len(clusters)):
		distance=dist(CHVector,clusters[i])
		if distance<minInd:
			ind=i
			minInd=distance
	return ind

#	Makes the BoVW feature vectors.
def makeBoVW(clusters,inputDir,outputDir):
	for contents in os.listdir(inputDir):
		contentName=os.path.join(inputDir,contents)
		if os.path.isdir(contentName):
			for filename in os.listdir(contentName):
				featureVector=[0 for i in range(BoVW_VectorLen)]
				infile=open(os.path.join(contentName,filename))
				outputFilename=os.path.join(outputDir,contents,filename)
				if not os.path.exists(os.path.dirname(outputFilename)):
					try:
						os.makedirs(os.path.dirname(outputFilename))
					except OSError as exc:
						if exc.errorno!=errorno.EEXIST:
							raise
				outfile=open(outputFilename,"w")
				for line in infile:
					number_strings=line.split()
					numbers=[float(n) for n in number_strings]
					clusterNum=classify(numbers,clusters)
					featureVector[clusterNum]+=1
				infile.close()
				for i in range(BoVW_VectorLen):
					outfile.write(str(featureVector[i])+" ")
				outfile.close()

#	Creates subdirectories if not present in a path.
def createPath(output):
	if not os.path.exists(os.path.dirname(output)):
		try:
			os.makedirs(os.path.dirname(output))
		except OSError as exc:
			if exc.errorno!=errorno.EEXIST:
				raise

#	Clubs data of all directories in 'direct' directory into file 'output'
def club(output,direct,ind):
	createPath(output)
	outputFile=open(output,"w")
	for contents in os.listdir(direct):
		if ind==1:
			contentName=os.path.join(direct,contents)
			if os.path.isdir(contentName):
				for filename in os.listdir(contentName):
					file=open(os.path.join(contentName,filename))
					outputFile.write(file.read())
		else:
			file=open(os.path.join(direct,contents))
			outputFile.write(file.read()+"\n")

#	Program starts here...
print ("\nThis program extracts Bag-of-Visual-Words feature vectors of images from their color histogram features.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for training input and output or default (o/d): ")

direct=""
directT=""
directOtrain=""
directOtest=""

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training images color histogram feature vectors directory for all classes: ")
	directT=raw_input("Enter the path (relative or complete) of the test images color histogram feature vectors directory for all classes: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store these BoVW feature vectors: ")
	dimension=input("Enter the dimension of the color histogram feature vectors present in these files: ")
	directOtrain=os.path.join(directO,"train")
	directOtest=os.path.join(directO,"test")
else:
	direct="../../data/Output/GMM/Dataset 2/B/featureVectorsCH/train"
	directT="../../data/Output/GMM/Dataset 2/B/featureVectorsCH/test"
	directOtrain="../../data/Output/GMM/Dataset 2/B/featureVectorsBoVW/train"
	directOtest="../../data/Output/GMM/Dataset 2/B/featureVectorsBoVW/test"
	dimension=24

if direct[len(direct)-1]!='/':
	direct+="/"
if directT[len(directT)-1]!='/':
	directT+="/"
if directOtrain[len(directOtrain)-1]!='/':
	directOtrain+="/"
if directOtest[len(directOtest)-1]!='/':
	directOtest+="/"

BoVW_VectorLen=input("Enter the number of clusters, you want to divide the data into: ")

print "Clubbing all feature vectors of all training images together..."
club(directOtrain+"train.txt",direct,1)

print "Clustering the training data into "+str(BoVW_VectorLen)+" clusters."

print "This may take a while, be patient..."
clusters=kMeansClustering(os.path.join(directOtrain,"train.txt"))

print "Done. Now making BoVW feature vectors of all images in training and test dataset."
makeBoVW(clusters,direct,directOtrain)
makeBoVW(clusters,directT,directOtest)

print "Clubbing image BoVW feature vectors of a class together..."
for contents in os.listdir(directOtrain):
	contentName=os.path.join(directOtrain,contents)
	if os.path.isdir(contentName) and contents!="use":
		createPath(os.path.join(directOtrain,"use",contents+".txt"))
		club(os.path.join(directOtrain,"use",contents+".txt"),contentName,2)

for contents in os.listdir(directOtest):
	contentName=os.path.join(directOtest,contents)
	if os.path.isdir(contentName) and contents!="use":
		createPath(os.path.join(directOtest,"use",contents+".txt"))
		club(os.path.join(directOtest,"use",contents+".txt"),contentName,2)
print "Everything done successfully."

#	End.