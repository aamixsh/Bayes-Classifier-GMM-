#	CS669 - Assignment 2 (Group-2) [24/10/17]
#	About: 
#		This program is for testing text data on the training models built using different number of clusters for GMM.

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random
				
dimension=24								#	Dimension of data vectors.
K=3											#	Value of K for making clusters in GMM.

clusterMeans=[]								#	Stores means of all clusters in all classes.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of all clusters of all classes.
clusterPi=[]								#	Stores mixing coefficients for all clusters of all classes.
testData=[]									#	Stores test data.
confusionMatClass=[]						#	Confusion matrices between 2 classes.
confusionMatrix=[]							#	Total confusion matrix for all classes.
testDataValues=[]

#	Reads input from test files.
def calcPrereqTest(direct):
	tempClassData=[]
	for filename in os.listdir(direct):
		file=open(os.path.join(direct,filename))
		data=[]
		for line in file:
			number_strings=line.split()
			numbers=[float(n) for n in number_strings]
			data.append(numbers)
		tempTestData=np.array(data)
		tempClassData.append(tempTestData)
		file.close()
	testData.append(tempClassData)

#	Generates likelihood values of data points for use in classifylikelihood function.
def preCalcLikelihood():
	for i in range(len(testData)):
		tempTestDataValues=[]
		for j in range(len(testData[i])):
			tempClassValues=[]
			for c in range(len(clusterMeans)):
				value=0
				for k in range(K):
					for y in range(len(testData[i][j])):
						tempValue=likelihood(testData[i][j][y],clusterMeans[c][k],clusterCovarianceMatrices[c][k])
						if tempValue<1e-300:
							tempValue=1e-300
						value+=math.log(clusterPi[c][k]*tempValue)
				tempClassValues.append(value)
			print tempClassValues
			tempTestDataValues.append(tempClassValues)
		testDataValues.append(tempTestDataValues)

#	Return the likelihood of a sample point 'x', given Gaussian parameters 'uK' and 'sigmaK'.
def likelihood(x,uK,sigmaK):
	value=1.0/((((2*math.pi)**(dimension))*(math.fabs(np.linalg.det(sigmaK))))**0.5)
	temp=[0 for i in range(dimension)]
	mul=0
	sigmaInvK=np.asmatrix(sigmaK).I.A
	for i in range(dimension):
		for j in range(dimension):
			temp[i]+=(x[j]-uK[j])*sigmaInvK[j][i]
	for i in range(dimension):
		mul+=temp[i]*(x[i]-uK[i])
	value*=math.exp(-0.5*mul)
	return value

#	Returns in the index of class with maximum likelihood of having the sample point 'x'.
def classifyLikelihood(i,j):
	# val=[0 for i in range(len(clusterMeans))]
	# for i in range(len(clusterMeans)):
	# 	for k in range(K):
	# 		for y in range(len(x)):
	# 			val[i]+=math.log(clusterPi[i][k]*likelihood(x[y],clusterMeans[i][k],clusterCovarianceMatrices[i][k]))
	# print val
	# print testDataValues[i][j]
	return np.argmax(testDataValues[i][j])

#	Calculates the confusion matrix of all classes together.
def calcConfusion():
	global confusionMatrix
	confusionMatrix=[[0 for i in range(len(clusterMeans))] for i in range(len(clusterMeans))]
	for i in range(len(testData)):
		for j in range(len(testData[i])):
			# x=testData[i][j]
			ret=classifyLikelihood(i,j)
			print ret
			confusionMatrix[ret][i]+=1

#	Calculates the confusion matrix of class with index 'ind' with respect to all other classes.
def calcConfusionClass(ind):
	temp=[[0 for i in range(2)] for j in range(2)]
	for j in range(len(testData)):
		for i in range(len(testData[j])):
			# x=testData[j][i]
			ret=classifyLikelihood(j,i)
			print ret
			if ind==j:
				if ret==ind:
					temp[0][0]+=1
				else:
					temp[1][0]+=1
			else: 
				if ret==ind:
					temp[0][1]+=1
				else:
					temp[1][1]+=1
	confusionMatClass.append(temp)

#	Program starts here...
print ("\nThis program is for testing text data of the color histogram feature vectors extracted from images and given GMM.\n")

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
	direct="../../data/Output/GMM/Dataset 2/B/featureVectorsCH/test/"
	directM="../../data/Output/GMM/Dataset 2/B/train_model_CH"
	directO="../../data/Output/GMM/Dataset 2/B/test_results_CH/"


if direct[len(direct)-1]!='/':
	direct+="/"
if directM[len(directM)-1]!='/':
	directM+="/"
if directO[len(directO)-1]!='/':
	directO+="/"

print "Reading input test files..."
for contents in os.listdir(direct):
	contentName=os.path.join(direct,contents)
	if os.path.isdir(contentName):
		calcPrereqTest(contentName)
print "Done."

for filename in os.listdir(directM):
	
	file=open(directM+filename)
	Input=file.readline()
	numbers=Input.split()
	inputFormat=[int(n) for n in numbers]
	
	dimension=inputFormat[0]
	K=inputFormat[1]
	numClasses=inputFormat[2]

	clusterMeans=[]
	clusterCovarianceMatrices=[]
	clusterPi=[]
	confusionMatClass=[]
	confusionMatrix=[]
	testDataValues=[]

	for n in range(numClasses):
		line=file.readline()
		numbers=line.split()
		tempClusterPi=[float(x) for x in numbers]
		clusterPi.append(tempClusterPi)

	for n in range(numClasses):
		tempClassMeans=[]
		for k in range(K):
			line=file.readline()
			numbers=line.split()
			tempClusterMeans=[float(x) for x in numbers]
			tempClassMeans.append(tempClusterMeans)
		clusterMeans.append(tempClassMeans)

	for n in range(numClasses):
		tempClassCovarianceMatrices=[]
		for k in range(K):
			tempClusterCovarianceMatrix=[]
			for i in range(dimension):
				line=file.readline()
				numbers=line.split()
				tempCovarianceMatrixRow=[float(x) for x in numbers]
				tempClusterCovarianceMatrix.append(tempCovarianceMatrixRow)
			clusterCovarianceMatrix=np.array(tempClusterCovarianceMatrix)
			tempClassCovarianceMatrices.append(clusterCovarianceMatrix)
		clusterCovarianceMatrices.append(tempClassCovarianceMatrices)	

	print "Testing data for K = "+str(K)+"."
	print "Calculating some prerequisties..."
	preCalcLikelihood()
	print "Done. Testing..."

	#	Calculating confusion matrices...
	for i in range(numClasses):
		calcConfusionClass(i)
	calcConfusion()
	
	print "Data testing complete. Writing results in files for future reference..."
	filev=open(directO+"values_k"+str(K)+".txt","w")
	filer=open(directO+"results_k"+str(K)+".txt","w")
	
	filer.write("The Confusion Matrix of all classes together is: \n")
	for i in range(numClasses):
		for j in range(numClasses):
			filev.write(str(confusionMatrix[i][j])+" ")
			filer.write(str(confusionMatrix[i][j])+" ")
		filev.write("\n")
		filer.write("\n")

	filer.write("\nThe Confusion Matrices for different classes are: \n")
	for i in range(len(confusionMatClass)):
		filer.write("\nClass "+str(i+1)+": \n")
		for x in range(2):
			for y in range(2):
				filev.write(str(confusionMatClass[i][x][y])+" ")
				filer.write(str(confusionMatClass[i][x][y])+" ")
			filev.write("\n")
			filer.write("\n")

	Accuracy=[]
	Precision=[]
	Recall=[]
	FMeasure=[]

	filer.write("\nDifferent quantitative values are listed below.\n")
	for i in range(numClasses):
		tp=confusionMatClass[i][0][0]
		fp=confusionMatClass[i][0][1]
		fn=confusionMatClass[i][1][0]
		tn=confusionMatClass[i][1][1]
		accuracy=float(tp+tn)/(tp+tn+fp+fn)
		precision=float(tp)/(tp+fp)
		recall=float(tp)/(tp+fn)
		fMeasure=2*precision*recall/(precision+recall)
		filer.write("\nClassification Accuracy for class "+str(i+1)+" is "+str(accuracy)+"\n")
		filer.write("Precision for class "+str(i+1)+" is "+str(precision)+"\n")
		filer.write("Recall for class "+str(i+1)+" is "+str(recall)+"\n")
		filer.write("F-measure for class "+str(i+1)+" is "+str(fMeasure)+"\n")
		filev.write(str(accuracy)+" "+str(precision)+" "+str(recall)+" "+str(fMeasure)+"\n")
		Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

	avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
	for i in range (numClasses):
		avgAccuracy+=Accuracy[i]
		avgPrecision+=Precision[i]
		avgRecall+=Recall[i]
		avgFMeasure+=FMeasure[i]
	avgAccuracy/=len(clusterMeans)
	avgPrecision/=len(clusterMeans)
	avgRecall/=len(clusterMeans)
	avgFMeasure/=len(clusterMeans)

	filer.write("\nAverage classification Accuracy is "+str(avgAccuracy)+"\n")
	filer.write("Average precision is "+str(avgPrecision)+"\n")
	filer.write("Average recall is "+str(avgRecall)+"\n")
	filer.write("Average F-measure is "+str(avgFMeasure)+"\n")
	filer.write("\n**End of results**")
	filev.write(str(avgAccuracy)+" "+str(avgPrecision)+" "+str(avgRecall)+" "+str(avgFMeasure)+"\n")
	filer.close()
	filev.close()
	
#	End.