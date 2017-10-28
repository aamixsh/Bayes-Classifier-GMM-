#	CS669 - Assignment 2 (Group-2) [24/10/17]
#	About: 
#		This program is for testing text data on the training models built using different number of clusters for GMM.

import numpy as np
import math
from PIL import Image
import os
import random
				
dimension=2									#	Dimension of data vectors.
K=3											#	Value of K for making clusters in GMM.

clusterMeans=[]								#	Stores means of all clusters in all classes.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of all clusters of all classes.
clusterPi=[]								#	Stores mixing coefficients for all clusters of all classes.
testData=[]									#	Stores test data.
imageSize=[]

#	Return the likelihood of a sample point 'x', given Gaussian parameters 'uK' and 'sigmaK'.
def likelihood(x,uK,sigmaK):
	Denom=((((2*math.pi)**(dimension))*(math.fabs(np.linalg.det(sigmaK))))**0.5)
	value=1.0/Denom
	temp=[0 for i in range(dimension)]
	mul=0
	sigmaInvK=np.asmatrix(sigmaK).I.A
	for i in range(dimension):
		for j in range(dimension):
			temp[i]+=(x[j]-uK[j])*sigmaInvK[j][i]
	for i in range(dimension):
		mul+=temp[i]*(x[i]-uK[i])
	if mul>1000:
		mul=1000
	elif mul<-1000:
		mul=-1000
	value*=math.exp(-0.5*mul)
	return value

#	Returns in the index of class with maximum likelihood of having the sample point 'x'.
def classifyLikelihood(x):
	val=[0 for i in range(K)]
	for k in range(K):
		val[k]=clusterPi[k]*likelihood(x,clusterMeans[k],clusterCovarianceMatrices[k])
	return np.argmax(val)

#	Tests the input test files and makes output images using training model.
def Test(direct,ind):
	outputData=""
	for k in range(len(testData[ind])):	
		ret=classifyLikelihood(testData[ind][k])
		if ret==0:
			outputData+=chr(255)+chr(255)+chr(255)
		elif ret==1:
			outputData+=chr(0)+chr(0)+chr(0)
		else:
			outputData+=chr(127)+chr(127)+chr(127)
	im=Image.frombytes("RGB",(imageSize[ind][0],imageSize[ind][1]),outputData)
	im.save(os.path.join(direct+filename)+str(ind)+".png","PNG")

#	Tests the input test files and makes output images using training model.
def calcPrereqTest(filename):
	file=open(filename)
	inputFS=file.readline()
	numbers=inputFS.split()
	imageS=[int(n) for n in numbers]
	imageSize.append(imageS)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	testData.append(np.array(data))	

#	Program starts here...
print ("\nThis program is for testing text data on the training model built with 3 clustered GMM.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for the training model data and test input/output or default (o/d): ")

direct=""
directM=""
directO=""

if(choice=='o'):
	directM=raw_input("Enter the path (relative or complete) of the training model data directory: ")
	direct=raw_input("Enter the path (relative or complete) of the test data directory: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store results after testing: ")
else:
	direct="../../data/Output/Clustering/Dataset 2/C/featureVectorsTest/"
	directM="../../data/Output/Clustering/Dataset 2/C/train_model/"
	directO="../../data/Output/Clustering/Dataset 2/C/test_results/"


if direct[len(direct)-1]!='/':
	direct+="/"
if directM[len(directM)-1]!='/':
	directM+="/"
if directO[len(directO)-1]!='/':
	directO+="/"

print "Calculating Prerequisties..."
for filename in os.listdir(direct):
	calcPrereqTest(direct+filename)
print "Done. Testing and creating output images..."

for filename in os.listdir(directM):
	
	file=open(directM+filename)

	Input=file.readline()
	numbers=Input.split()
	inputFormat=[int(n) for n in numbers]
	dimension=inputFormat[0]
	K=inputFormat[1]

	clusterMeans=[]
	clusterCovarianceMatrices=[]
	clusterPi=[]
	confusionMatClass=[]
	confusionMatrix=[]

	line=file.readline()
	numbers=line.split()
	for x in numbers:
		clusterPi.append(float(x))

	for k in range(K):
		line=file.readline()
		numbers=line.split()
		tempClusterMeans=[float(x) for x in numbers]
		clusterMeans.append(tempClusterMeans)

	for k in range(K):
		tempClusterCovarianceMatrix=[]
		for i in range(dimension):
			line=file.readline()
			numbers=line.split()
			tempCovarianceMatrixRow=[float(x) for x in numbers]
			tempClusterCovarianceMatrix.append(tempCovarianceMatrixRow)
		clusterCovarianceMatrix=np.array(tempClusterCovarianceMatrix)
		clusterCovarianceMatrices.append(clusterCovarianceMatrix)	

	print "Testing data for file "+filename+"..."
	if filename=="gmm_k3_energy_267185.508057.txt":	
		for i in range(3):
			Test(directO,i)

	# #	Calculating confusion matrices...
	# for i in range(numClasses):
	# 	calcConfusionClass(i)
	# calcConfusion()
	
	# print "Data testing complete. Writing results in files for future reference..."
	# filev=open(directO+"values_k"+str(K)+".txt","w")
	# filer=open(directO+"results_k"+str(K)+".txt","w")
	
	# filer.write("The Confusion Matrix of all classes together is: \n")
	# for i in range(numClasses):
	# 	for j in range(numClasses):
	# 		filev.write(str(confusionMatrix[i][j])+" ")
	# 		filer.write(str(confusionMatrix[i][j])+" ")
	# 	filev.write("\n")
	# 	filer.write("\n")

	# filer.write("\nThe Confusion Matrices for different classes are: \n")
	# for i in range(len(confusionMatClass)):
	# 	filer.write("\nClass "+str(i+1)+": \n")
	# 	for x in range(2):
	# 		for y in range(2):
	# 			filev.write(str(confusionMatClass[i][x][y])+" ")
	# 			filer.write(str(confusionMatClass[i][x][y])+" ")
	# 		filev.write("\n")
	# 		filer.write("\n")

	# Accuracy=[]
	# Precision=[]
	# Recall=[]
	# FMeasure=[]

	# filer.write("\nDifferent quantitative values are listed below.\n")
	# for i in range(numClasses):
	# 	tp=confusionMatClass[i][0][0]
	# 	fp=confusionMatClass[i][0][1]
	# 	fn=confusionMatClass[i][1][0]
	# 	tn=confusionMatClass[i][1][1]
	# 	accuracy=float(tp+tn)/(tp+tn+fp+fn)
	# 	precision=float(tp)/(tp+fp)
	# 	recall=float(tp)/(tp+fn)
	# 	fMeasure=2*precision*recall/(precision+recall)
	# 	filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
	# 	filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
	# 	filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
	# 	filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
	# 	filev.write(str(accuracy)+" "+str(precision)+" "+str(recall)+" "+str(fMeasure)+"\n")
	# 	Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

	# avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
	# for i in range (numClasses):
	# 	avgAccuracy+=Accuracy[i]
	# 	avgPrecision+=Precision[i]
	# 	avgRecall+=Recall[i]
	# 	avgFMeasure+=FMeasure[i]
	# avgAccuracy/=len(clusterMeans)
	# avgPrecision/=len(clusterMeans)
	# avgRecall/=len(clusterMeans)
	# avgFMeasure/=len(clusterMeans)

	# filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
	# filer.write("Average precision is "+str(avgPrecision)+"\n")
	# filer.write("Average recall is "+str(avgRecall)+"\n")
	# filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
	# filer.write("\n**End of results**")
	# filev.write(str(avgAccuracy)+" "+str(avgPrecision)+" "+str(avgRecall)+" "+str(avgFMeasure)+"\n")
	# filer.close()
	# filev.close()
	
#	End.