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

direct=""
directO=""

#	Return the likelihood of a sample point 'x', given Gaussian parameters 'uK' and 'sigmaK'.
def likelihood(x,uK,sigmaK):
	Denom=((((2*math.pi)**(dimension))*(math.fabs(np.linalg.det(sigmaK))))**0.5)
	if Denom==0:
		Denom=1e-300
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
	global clusters, clusterMeans, clusterCovarianceMatrices, clusterPi
	file=open(filename)
	data=[]
	for line in file:
		number_strings=line.split()
		numbers=[float(n) for n in number_strings]
		data.append(numbers)
	tempClass=np.array(data)
	N=len(tempClass)
	del data
	file.close()
	print "done"
	#	K-means clustering for initiating GMM formation...

	#	Assigning random means to the K clusters...
	tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
	randomKMeans=random.sample(range(0,N-1),K)
	for i in range(K):
		for j in range(dimension):
			tempClusterMean[i][j]=tempClass[randomKMeans[i]][j]

	print tempClusterMean
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
	
	clusters.append(tempClusters)
	clusterMeans.append(tempClusterMean)
	
	tempClassInd=len(clusters)-1
	tempClassSize=N

	calcCovarianceMat(tempClassInd)

	tempClusterPi=[]
	for i in range(K):
		print len(tempClusters[i])
		tempClusterPi.append(float(len(tempClusters[i]))/N)

	clusterPi.append(tempClusterPi)

	file=open(directO+"init_k"+str(K)+".txt","w")
	file.write(str(dimension)+" "+str(K)+" "+str(len(clusters))+"\n")
	for i in range(len(clusters)):
		for k in range(K):
			file.write(str(clusterPi[i][k])+" ")
		file.write("\n")
	for i in range(len(clusters)):
		for k in range(K):
			for j in range(dimension):
				file.write(str(clusterMeans[i][k][j])+" ")
			file.write("\n")
	for i in range(len(clusters)):
		for k in range(K):
			for j in range(dimension):
				for l in range(dimension):
					file.write(str(clusterCovarianceMatrices[i][k][j][l])+" ")
				file.write("\n")
	file.close()

	#	Re-evaluating centres until the energy of changes becomes insignificant (convergence)...
	while energy>1000:
		tempClusterMean=[[0 for i in range(dimension)] for i in range(K)]
		for i in range(K):
			for j in range(len(tempClusters[i])):
				for k in range(dimension):
					tempClusterMean[i][k]+=tempClusters[i][j][k]
			for k in range(dimension):
				tempClusterMean[i][k]/=len(tempClusters[i])
		del tempClusters
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
		print energy
	
	clusters[tempClassInd]=tempClusters
	clusterMeans[tempClassInd]=tempClusterMean

	#	Calculating Covariance Matrices for all clusters...
	calcCovarianceMat(tempClassInd)
	
	#	Calculating mixing coefficients for all clusters...
	for i in range(K):
		print len(clusters[tempClassInd][i])
		clusterPi[tempClassInd][k]=(float(len(clusters[tempClassInd][i]))/N)

	file=open(directO+"after_K_Means_k"+str(K)+".txt","w")
	file.write(str(dimension)+" "+str(K)+" "+str(len(clusters))+"\n")
	for i in range(len(clusters)):
		for k in range(K):
			file.write(str(clusterPi[i][k])+" ")
		file.write("\n")
	for i in range(len(clusters)):
		for k in range(K):
			for j in range(dimension):
				file.write(str(clusterMeans[i][k][j])+" ")
			file.write("\n")
	for i in range(len(clusters)):
		for k in range(K):
			for j in range(dimension):
				for l in range(dimension):
					file.write(str(clusterCovarianceMatrices[i][k][j][l])+" ")
				file.write("\n")
	file.close()
	del clusters

	#	Gaussian Mixture Modelling...

	#	Using these initial calculated values for the EM algorithm.
	
	tempClusterCovarianceMatrices=clusterCovarianceMatrices[tempClassInd]
	energy=np.inf
	tempL=0
	print "lol"

	while energy>5000:
		
		#	Expectation step in the algorithm...
		tempGammaZ=[[0 for i in range (K)] for j in range (N)]
		tempLikelihoodTerms=[[0 for i in range(K)] for j in range(N)]
		tempDenom=[0 for i in range(N)]
		tempGammaSum=[0 for i in range(K)]
		newTempL=0

		#	Calculating responsibilty terms using previous values of parameters. 
		for n in range(N):
			for k in range(K):
				determinant=np.linalg.det(tempClusterCovarianceMatrices[k])
				while determinant==0:
					print tempClusterCovarianceMatrices[k]
					for i in range(dimension):
						tempClusterCovarianceMatrices[k][i][i]+=0.001
					determinant=np.linalg.det(tempClusterCovarianceMatrices[k])
				varLikelihood=likelihood(tempClass[n],tempClusterMean[k],tempClusterCovarianceMatrices[k])
				if varLikelihood==0:
					varLikelihood=1e-300
				tempLikelihoodTerms[n][k]=tempClusterPi[k]*varLikelihood
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
			tempMatrix/=tempGammaSum[k]
			determinant=np.linalg.det(tempMatrix)
			while determinant==0:
				for i in range(dimension):
					tempMatrix[i][i]+=1
				determinant=np.linalg.det(tempMatrix)
			if tempL==0:
				tempClusterCovarianceMatrices.append(tempMatrix)
			else:
				tempClusterCovarianceMatrices[k]=tempMatrix

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

		file=open(directO+"gmm_k"+str(K)+"_energy_"+str(energy)+".txt","w")
		file.write(str(dimension)+" "+str(K)+" "+str(1)+"\n")
		for i in range(1):
			for k in range(K):
				file.write(str(tempClusterPi[k])+" ")
			file.write("\n")
		for i in range(1):
			for k in range(K):
				for j in range(dimension):
					file.write(str(tempClusterMean[k][j])+" ")
				file.write("\n")
		for i in range(1):
			for k in range(K):
				for j in range(dimension):
					for l in range(dimension):
						file.write(str(tempClusterCovarianceMatrices[k][j][l])+" ")
					file.write("\n")
		file.close()

		print energy

	clusterMeans[tempClassInd]=tempClusterMean
	clusterCovarianceMatrices[tempClassInd]=tempClusterCovarianceMatrices
	clusterPi.append(tempClusterPi)

#	Program starts here...
print ("\nThis program trains text data extracted from feature vectors of cell images and building GMM using 3 xlusters.\n")

#	Parsing Input... 
choice= raw_input("Do you want to use your own directory for training input and output or default (o/d): ")

if(choice=='o'):
	direct=raw_input("Enter the path (relative or complete) of the training data directory: ")
	directO=raw_input("Enter the path (relative or complete) of the directory to store these feature vectors: ")
else:
	direct="../../data/Output/Clustering/Dataset 2/C/featureVectorsTrain"
	directO="../../data/Output/Clustering/Dataset 2/C/train_model/train_model_10"

if direct[len(direct)-1]!='/':
	direct+="/"
if directO[len(directO)-1]!='/':
	directO+="/"

print "Training data for K = "+str(K)+"..."
for filename in os.listdir(direct):
	if filename=="trainingFeatures10.txt":
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
