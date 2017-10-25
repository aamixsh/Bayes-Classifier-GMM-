#	CS669 - Assignment 2 (Group-2) [24/10/17]
#	About: 
#		This program is for training text data and build GMM parameters for it using different number of clusters.

import numpy as np
import math
import os
import random
			
dimension=2									#	Dimension of data vectors.
K=3											#	Value of K for making clusters in GMM.

clusters=[]									#	Stores training data in form of clusters under every class for future reference. 
clusterMeans=[]								#	Stores means of all clusters in all classes.
clusterCovarianceMatrices=[]				#	Stores covariance matrices of all clusters of all classes.
clusterPi=[]								#	Stores mixing coefficients for all clusters of all classes.

#	Return the likelihood of a sample point 'x', given Gaussian parameters 'uK' and 'sigmaK'.
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

#	Returns the covaricance between dimension 'i' and 'j', of 'cluster' indexed cluster in class with index 'ind'.
def Covariance(ind,cluster,i,j):
	sum=0
	for k in range(len(clusters[ind][cluster])):
		x=clusters[ind][cluster][k]
		sum+=(x[i]-clusterMeans[ind][cluster][i])*(x[j]-clusterMeans[ind][cluster][j])
	sum/=len(clusters[ind][cluster])
	return sum

#	Calculates covariance matrices of all clusters in class with index 'ind'.
def calcCovarianceMat(ind):
	tempClusterCovarianceMatrices=[]
	for i in range(K):
		tempCovarianceMat=[[0 for k in range(dimension)] for j in range(dimension)]
		for j in range(dimension):
			for k in range(dimension):
				tempCovarianceMat[j][k]=Covariance(ind,i,j,k)
		tempClusterCovarianceMatrices.append(tempCovarianceMat)
	clusterCovarianceMatrices.append(tempClusterCovarianceMatrices)

#	Calculates distance between two points in 'dimension' dimensional space.
def dist(x,y):
	distance=0
	for i in range(dimension):
		distance+=(x[i]-y[i])**2
	distance=math.sqrt(distance)
	return (distance)

#	Function to calculate the paramters of the training model using data in file "filename".
def calcPrereqTrain(filename):
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
	while energy>0.000001:
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
		tempGammaZ=[[0 for i in range (K)] for j in range (N)]
		tempLikelihoodTerms=[[0 for i in range(K)] for j in range(N)]
		tempDenom=[0 for i in range(N)]
		tempGammaSum=[0 for i in range(K)]
		newTempL=0

		#	Calculating responsibilty terms using previous values of parameters. 
		for n in range(N):
			for k in range(K):
				tempLikelihoodTerms[n][k]=tempClusterPi[k]*likelihood(tempClass[n],tempClusterMean[k],tempClusterCovarianceMatrices[k])
				tempDenom[n]+=tempLikelihoodTerms[n][k]
			for k in range(K):
				tempGammaZ[n][k]=tempLikelihoodTerms[n][k]/tempDenom[n]
				tempGammaSum[k]+=tempGammaZ[n][k]

		#	Maximization step in the algorithm...
		#	Refining mean vectors.
		for k in range(K):
			for i in range(dimension):
				tempClusterMean[k][i]=0
				for n in range(N):
					tempClusterMean[k][i]+=tempGammaZ[n][k]*tempClass[n][i]
				tempClusterMean[k][i]/=tempGammaSum[k]

		#	Refining covariance matrices.
		for k in range(K):
			tempMatrix=[[0 for i in range(dimension)] for j in range(dimension)]
			for n in range(N):
				tempMatrix+=tempGammaZ[n][k]*np.outer((tempClass[n]-tempClusterMean[k]),(tempClass[n]-tempClusterMean[k]))
			if tempL==0:
				tempClusterCovarianceMatrices.append(tempMatrix/tempGammaSum[k])
			else:
				tempClusterCovarianceMatrices[k]=tempMatrix/tempGammaSum[k]

		#	Refining mixing coefficients.
		for k in range(K):
			tempClusterPi[k]=tempGammaSum[k]/N

		for n in range(N):
			newTempL+=math.log(tempDenom[n])

		if tempL==0:
			tempL=newTempL
			continue
		else:
			energy=math.fabs(tempL-newTempL)
			tempL=newTempL

	clusterMeans[tempClassInd]=tempClusterMean
	clusterCovarianceMatrices[tempClassInd]=tempClusterCovarianceMatrices
	clusterPi.append(tempClusterPi)

#	Program starts here...
print ("\nThis program is for training text data and build GMM parameters for it using different number of clusters.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for training input and output or default (o/d): ")

direct=""
directO=""
choiceIn=1

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training data directory: ")
	inpDim=input("Enter the number of dimensions in the data (for input format, refer README.txt): ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store parameters of the training model: ")
	dimension=inpDim
else:
	choiceIn=input("Dataset (1/2): ")
	if choiceIn==1:
		direct="../../data/Input/GMM/Dataset 1/train"
		directO="../../data/Output/GMM/Dataset 1/train_model/"
	else:
		direct="../../data/Input/GMM/Dataset 2/A/train"
		directO="../../data/Output/GMM/Dataset 2/A/train_model/"


if direct[len(direct)-1]!='/':
	direct+="/"
if directO[len(directO)-1]!='/':
	directO+="/"

print "Enter the value of K upto which you want to generate training models."
maxK=input("Beware, large K's can result in singularity problems: ")

for k in range(maxK):
	
	clusters=[]
	clusterMeans=[]
	clusterCovarianceMatrices=[]
	clusterPi=[]
	K=k+1

	print "Training data for K = "+str(K)+"..."
	for filename in os.listdir(direct):
		calcPrereqTrain(direct+filename)

	print "Data training complete. Writing results in a file for future use..."
	file=open(directO+"k"+str(K)+".txt","w")
	file.write(str(dimension)+" "+str(K)+" "+str(len(clusters))+"\n")
	for i in range(len(clusterPi)):
		for k in range(K):
			file.write(str(clusterPi[i][k])+" ")
		file.write("\n")
	for i in range(len(clusterMeans)):
		for k in range(K):
			for j in range(dimension):
				file.write(str(clusterMeans[i][k][j])+" ")
			file.write("\n")
	for i in range(len(clusterCovarianceMatrices)):
		for k in range(K):
			for j in range(dimension):
				for l in range(dimension):
					file.write(str(clusterCovarianceMatrices[i][k][j][l])+" ")
				file.write("\n")
	file.close()

#	End.