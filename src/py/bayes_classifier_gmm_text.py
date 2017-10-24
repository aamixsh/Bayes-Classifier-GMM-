#	CS669 - Assignment 2 (Group-2) [17/10/17]
#	About: 
#		This program classifies the data for different classes already given in text using GMM.

import numpy as np
import math
import matplotlib.pyplot as plt
import os
import random

"""/*




*/"""
testData=[]									#	Stores test data.				
dimension=2									#	Dimension of data vectors.
K=4											#	Value of K for making clusters in GMM.

clusters=[]									#	Stores training data in form of clusters under every class for future reference. 
clusterMeans=[]								#	Stores means of all clusters in all classes.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of all clusters of all classes.
clusterPi=[]								#	Stores mixing coefficients for all clusters of all classes.

confusionMatClass=[]						#	Confusion matrices between 2 classes.
confusionMatrix=[]							#	Total confusion matrix for all classes.
# covarianceMatrix=np.zeros(shape=(dimension,dimension))
# covarianceMatrixInv=np.zeros(shape=(dimension,dimension))
# average_variance=0

def calcPrereqTest(filename):
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempTestData=np.array(data)
	testData.append(tempTestData)

def likelihood(x,uK,sigmaK):
	value=1.0/(((2*math.pi)**(dimension))*((np.linalg.det(sigmaK)))**0.5)
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

def expect(ind,cluster,i,j):
	sum=0
	for k in range(len(clusters[ind][cluster])):
		x=clusters[ind][cluster][k]
		sum+=(x[i]-clusterMeans[ind][cluster][i])*(x[j]-clusterMeans[ind][cluster][j])
	sum/=len(clusters[ind][cluster])
	return sum

def calcCovarianceMat(ind):
	tempClusterCovarianceMatrices=[]
	for i in range(K):
		tempCovarianceMat=[[0 for k in range(dimension)] for j in range(dimension)]
		for j in range(dimension):
			for k in range(dimension):
				tempCovarianceMat[j][k]=expect(ind,i,j,k)
		tempClusterCovarianceMatrices.append(tempCovarianceMat)
	clusterCovarianceMatrices.append(tempClusterCovarianceMatrices)

def dist(x,y):
	distance=0
	for i in range(dimension):
		distance+=(x[i]-y[i])**2
	distance=math.sqrt(distance)
	return (distance)

def calcPrereqTrain(filename):
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)

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
	
	#	Re-evaluating centres until the energy of changes becomes insignificant...
	while energy>0.000001:
		print energy
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
		energy=math.fabs(totDistance-newTotDistance);
		totDistance=newTotDistance;

	print energy

	clusters.append(tempClusters)
	clusterMeans.append(tempClusterMean)
	
	tempClassInd=len(clusters)-1
	tempClassSize=N
	
	#	Calculating Covariance Matrices for all clusters...
	calcCovarianceMat(tempClassInd)
	
	#	Calculating mixing coefficients for all clusters...
	tempClusterPi=[]
	for i in range(K):
		tempClusterPi.append(float(len(tempClusters[i]))/N)

	#	Gaussian Mixture Modelling...

	#	Using these initial calculated values for the EM algorithm.
	
	tempClusterCovarianceMatrices=clusterCovarianceMatrices[tempClassInd]
	energy=np.inf
	tempL=0

	while energy>0.001:
		#	Expectation step in the algorithm...

		print energy
		tempGammaZ=[[0 for i in range (K)] for j in range (N)]
		tempLikelihoodTerms=[[0 for i in range(K)] for j in range(N)]
		tempDenom=[0 for i in range(N)]
		tempGammaSum=[0 for i in range(K)]
		newTempL=0

		for n in range(N):
			for k in range(K):
				tempLikelihoodTerms[n][k]=tempClusterPi[k]*likelihood(tempClass[n],tempClusterMean[k],tempClusterCovarianceMatrices[k])
				tempDenom[n]+=tempLikelihoodTerms[n][k]
			for k in range(K):
				tempGammaZ[n][k]=tempLikelihoodTerms[n][k]/tempDenom[n]
				tempGammaSum[k]+=tempGammaZ[n][k]

		#	Maximization step in the algorithm...
		for k in range(K):
			for i in range(dimension):
				tempClusterMean[k][i]=0
				for n in range(N):
					tempClusterMean[k][i]+=tempGammaZ[n][k]*tempClass[n][i]
				tempClusterMean[k][i]/=tempGammaSum[k]

		for k in range(K):
			tempMatrix=[[0 for i in range(dimension)] for j in range(dimension)]
			for n in range(N):
				tempMatrix+=tempGammaZ[n][k]*np.outer((tempClass[n]-tempClusterMean[k]),(tempClass[n]-tempClusterMean[k]))
			if tempL==0:
				tempClusterCovarianceMatrices.append(tempMatrix/tempGammaSum[k])
			else:
				tempClusterCovarianceMatrices[k]=tempMatrix/tempGammaSum[k]

		for n in range(N):
			newTempL+=math.log(tempDenom[n])

		for k in range(K):
			tempClusterPi[k]=tempGammaSum[k]/N

		if tempL==0:
			tempL=newTempL
			continue
		else:
			energy=math.fabs(tempL-newTempL)
			tempL=newTempL

	clusterMeans[tempClassInd]=tempClusterMean
	clusterCovarianceMatrices[tempClassInd]=tempClusterCovarianceMatrices
	clusterPi.append(tempClusterPi)

# def classVal(x,ind):
# 	first_term=0
# 	tempFirstTerm=np.zeros(shape=(1,dimension))
# 	for j in range(dimension):
# 		for k in range(dimension):
# 			tempFirstTerm[0,j]+=x[k]*covarianceMatricesInv[ind][k,j]
# 	for j in range(dimension):
# 		first_term+=tempFirstTerm[0,j]*x[j]
# 	first_term*=-0.5
# 	second_term=0
# 	tempSecondTerm=np.zeros(shape=(dimension,1))
# 	for j in range(dimension):
# 		for k in range(dimension):
# 			tempSecondTerm[j,0]+=covarianceMatricesInv[ind][j,k]*mean[ind][k]
# 	third_term=0
# 	for j in range(dimension):
# 		second_term+=tempSecondTerm[j,0]*x[j]
# 		third_term+=tempSecondTerm[j,0]*mean[ind][j]
# 	third_term+math.log(np.linalg.det(covarianceMatrices[ind]))
# 	third_term*=-0.5
# 	tot=0
# 	for j in range(len(classes)):
# 		tot+=len(classes[j])
# 	third_term+=math.log(float(len(classes[ind]))/tot)
# 	return first_term+second_term+third_term	

# def g(x,first,second):
# 	if classVal(x,first)-classVal(x,second)<0:
# 		return first
# 	else:
# 		return second

def classifyLikelihood(x):
	# print x
	val=[0 for i in range(len(clusters))]
	for i in range(len(clusters)):
		for k in range(K):
			val[i]+=clusterPi[i][k]*likelihood(x,clusterMeans[i][k],clusterCovarianceMatrices[i][k])
	# print (val)
	# tempo=input()
	return np.argmax(val)

def calcConfusion():
	global confusionMatrix
	confusionMatrix=[[0 for i in range(len(clusters))] for i in range(len(clusters))]
	for i in range(len(testData)):
		for j in range(len(testData[i])):
			x=testData[i][j]
			ret=classifyLikelihood(x)
			confusionMatrix[ret][i]+=1

def calcConfusionClass(ind):
	temp=[[0 for i in range(2)] for j in range(2)]
	for j in range(len(testData)):
		for i in range(len(testData[j])):
			x=testData[j][i]
			ret=classifyLikelihood(x)
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

print ("\nThis program is a Bayes Classifier built using GMM for text data.\n")

#	Parsing Input... 

choice= raw_input("Do you want to use your own directory for training input or default (o/d): ")

direct=""
choiceIn=1

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training data directory : ")
else:
	choiceIn=input("Dataset (1/2): ")
	if choiceIn==1:
		direct="../../data/Input/GMM/Dataset 1/train";
	else:
		direct="../../data/Input/GMM/Dataset 2/A/train";

if direct[len(direct)-1]!='/':
	direct+="/";

for filename in os.listdir(direct):
	calcPrereqTrain(direct+filename)

print clusterMeans
print clusterPi


if choice=='o':
	direct=raw_input("Enter the path (relative or complete) of the test data directory : ")	
else:
	if choiceIn==1:
		direct="../../data/Input/GMM/Dataset 1/test";
	else:
		direct="../../data/Input/GMM/Dataset 2/A/test";

if direct[len(direct)-1]!='/':
	direct+="/";

for filename in os.listdir(direct):
	calcPrereqTest(direct+filename)

#	Using various values of k for finding the optimum, which best classifies the data with reasonable computation.



# choices=['ls','nl','rd']

# for i in range(len(classes)):
# 	for j in range(dimension):
# 		average_variance+=variance[i][j]
# average_variance/=len(classes)*dimension

# covarianceMatrix=average_variance*np.identity(dimension)
# covarianceMatrixInv=np.asmatrix(covarianceMatrix).I

# print "\nThe average variance calculated for all classes comes out to be",average_variance

# print "\nThe mean and variance vectors for different classes are: \n"
# for i in range(len(mean)):
# 	print "Class ",i+1,": Mean - ",mean[i]," Var - ",variance[i]

for i in range(len(clusters)):
	print clusterMeans[i]

for i in range(len(clusters)):
	calcConfusionClass(i)
print "\n"
calcConfusion()

print confusionMatrix
print confusionMatClass


Accuracy=[]
Precision=[]
Recall=[]
FMeasure=[]

print "\nThe Confusion Matrices for different classes are: "
for i in range(len(clusters)):
	print "\nConfusion Matrix for class",i+1,": \n"
	print np.asmatrix(confusionMatClass[i])
	tp=confusionMatClass[i][0][0]
	fp=confusionMatClass[i][0][1]
	fn=confusionMatClass[i][1][0]
	tn=confusionMatClass[i][1][1]
	accuracy=float(tp+tn)/(tp+tn+fp+fn)
	precision=float(tp)/(tp+fp)
	recall=float(tp)/(tp+fn)
	fMeasure=2*precision*recall/(precision+recall)
	print "\nClassification Accuracy for class",i+1,"is",accuracy
	print "Precision for class",i+1,"is",precision
	print "Recall for class",i+1,"is",recall
	print "F-measure for class",i+1,"is",fMeasure
	Accuracy.append(accuracy),Precision.append(precision),Recall.append(recall),FMeasure.append(fMeasure)

avgAccuracy,avgPrecision,avgRecall,avgFMeasure=0,0,0,0
for i in range (len(clusters)):
	avgAccuracy+=Accuracy[i]
	avgPrecision+=Precision[i]
	avgRecall+=Recall[i]
	avgFMeasure+=FMeasure[i]
avgAccuracy/=len(clusters)
avgPrecision/=len(clusters)
avgRecall/=len(clusters)
avgFMeasure/=len(clusters)

print "\nThe Confusion Matrix of all classes together is: \n"
print np.asmatrix(confusionMatrix)
print "\nAverage classification Accuracy is",avgAccuracy
print "Average precision is",avgPrecision
print "Average recall is",avgRecall
print "Average F-measure is",avgFMeasure

# print "\nPlease wait for a minute or two while the program generates graphs..."

# colors=['b','g','r']
# colorsTestData=['c','m','y']

# l=1
# f=[]

# f.append(plt.figure(l))
# l+=1
# minArr=[0 for i in range(dimension)]
# maxArr=[0 for i in range(dimension)]
# for i in range(dimension):
# 	minArr[i]=classesRange[0][0][i]
# 	maxArr[i]=classesRange[0][1][i]

# for i in range(len(classesRange)):
# 	for j in range(dimension):
# 		if(minArr[j]>classesRange[i][0][j]):
# 			minArr[j]=classesRange[i][0][j]
# 		if(maxArr[j]<classesRange[i][1][j]):
# 			maxArr[j]=classesRange[i][1][j]

# plt.subplot(111)
# xRange=np.arange(minArr[0],maxArr[0],float(maxArr[0]-minArr[0])/100)
# yRange=np.arange(minArr[1],maxArr[1],float(maxArr[1]-minArr[1])/100)
# for i in range(len(xRange)):
# 	for j in range(len(yRange)):
# 		X=[0,0]
# 		X[0]=xRange[i];
# 		X[1]=yRange[j];
# 		plt.plot(xRange[i],yRange[j],'.',color=colors[gi(X)])
# for j in range(len(classes)):
# 	plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'o',color=colorsTestData[j],label='Class {i}'.format(i=j))
# f[l-2].suptitle("Decision Region plot for all Classes")
# f[l-2].savefig('../../data/Output/A_AllClasses_DR_'+choices[choice-1]+'.png')


# for j in range(len(classes)):
# 	for k in range(j+1,len(classes)):
# 		f.append(plt.figure(l))
# 		l+=1
# 		minArr=[0 for i in range(dimension)]
# 		maxArr=[0 for i in range(dimension)]
# 		for i in range(dimension):
# 			minArr[i]=classesRange[j][0][i]
# 			maxArr[i]=classesRange[j][1][i]
# 		for i in range(dimension):
# 			if(minArr[i]>classesRange[k][0][i]):
# 				minArr[i]=classesRange[k][0][i]
# 			if(maxArr[i]<classesRange[k][1][i]):
# 				maxArr[i]=classesRange[k][1][i]
# 		plt.subplot(111)
# 		xRange=np.arange(minArr[0],maxArr[0],float(maxArr[0]-minArr[0])/100)
# 		yRange=np.arange(minArr[1],maxArr[1],float(maxArr[1]-minArr[1])/100)
# 		for m in range(len(xRange)):
# 			for n in range(len(yRange)):
# 				X=[0,0]
# 				X[0]=xRange[m];
# 				X[1]=yRange[n];
# 				plt.plot(xRange[m],yRange[n],'.',color=colors[g(X,j,k)])
# 			plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'o',color=colorsTestData[j],label='Class {i}'.format(i=j))
# 			plt.plot([classes[k][i][0] for i in range(len(classes[k]))],[classes[k][i][1] for i in range(len(classes[k]))],'o',color=colorsTestData[k],label='Class {i}'.format(i=k))
# 		label="Decision Region plot for class pair ("+str(j+1)+","+str(k+1)+")"
# 		f[l-2].suptitle(label)
# 		f[l-2].savefig('../../data/Output/A_ClassPair_'+str(j+1)+'_'+str(k+1)+'_DR_'+choices[choice-1]+'.png')

# for i in range(len(f)):
# 	f[i].show()

# g=plt.figure(5)
# for j in range(len(classes)):
# 	ax=plt.subplot(111)
# 	plt.plot([classes[j][i][0] for i in range(len(classes[j]))],[classes[j][i][1] for i in range(len(classes[j]))],'.',color=colors[j],label='Class {i}'.format(i=j))
# 	u=[]
# 	for k in range(dimension):
# 		tempU=np.linspace(classesRange[j][0][k],classesRange[j][1][k],10)
# 		u.append(tempU)
# 	x,y=np.meshgrid(u[0],u[1]) 
# 	temp=-0.5*covarianceMatrixInv
# 	temp1=np.matmul(covarianceMatrixInv,mean[j])
# 	const=np.matmul(np.matmul(mean[j].transpose(),temp),mean[j])
# 	tot=0
# 	for j in range(len(classes)):
# 		tot+=len(classes[j])
# 	constant=const[0,0]-0.5*math.log(np.linalg.det(covarianceMatrix))+math.log(float(len(classes[j]))/tot)
# 	z=(temp[0,0])*(x**2)+2*(temp[0,1])*x*y+temp[1,1]*(y**2)+temp1[0,0]*x+temp1[0,1]*y+constant
# 	ax.contour(x,y,z)

# g.suptitle("Constant Density Contours for all classes")
# g.savefig('../../data/Output/A_AllClasses_CDC_'+choices[choice-1]+'.png')
# plt.axis('scaled')
# g.show()

# plt.show()

