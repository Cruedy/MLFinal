#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SF Function: Scoring the Cluster Structure
# Based on the following paper
# Intelligent Data Analysis 12 (2008) 529–548 IOS Press
# A comprehensive validity index for clustering, S. Saitta,∗, B. Raphael and I.F.C. Smith
# Use: NEUR182_SF_Function(InData,NClusters,max_iter);

# Function to calculate the SFScore using kmeans
def SFScore_kmeans(InData,kval):
    # Will return WCD, BCD, SF, and SI
    # Saitta et al. (2008): Intelligent Data Analysis 12 (2008) 529–548 529
    # "A comprehensive validity index for clustering"
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import scale
    from sklearn.metrics import silhouette_score as SI
    from numpy.matlib import repmat
    import numpy as np
    
    NValues = InData.shape[0]
    scaleData = scale(InData,axis=0)
    
    sk_KMeans = KMeans(n_clusters = kval,
                       init = 'k-means++',
                       n_init = 1,
                       max_iter = 1000,
                       tol = 1e-10,
                       verbose = 0,
                       algorithm = "lloyd")
    sk_KMeans.fit(scaleData)
    pred_classes_sk = sk_KMeans.labels_
    sk_centroids = sk_KMeans.cluster_centers_
    
    # The code below is not efficient. However, it breaks down the steps so that they should be comprehensible.
    # If you found that your own code repeatedly gave you values of Infinity, then
    # you did not fully comprehend the paper's equations and you were not able to
    # successfully translate the equations into ML code
    # Please go through this code and see where you were making mistakes.
    
    # Calculate within cluster distance (WCD)
    # Saitta et al. (2008): Intelligent Data Analysis 12 (2008) 529–548 529
    # "A comprehensive validity index for clustering"
    # Equation 11
    InClust_WCD = np.empty(shape=(kval,1))
    AllNValuesInCluster = np.empty(shape=(kval,1),dtype=int)
    for clust_num in range(0,kval):
        clust_indices = np.where(pred_classes_sk == clust_num)
        clust_data = np.squeeze(scaleData[clust_indices,:])
        # Keep in mind Features go across columns, cases down rows
        NValuesInCluster = np.size(clust_indices)
        RepMat = repmat(sk_centroids[clust_num,:],NValuesInCluster,1)
        InClust_WCD[clust_num] = np.sqrt((1/NValuesInCluster) * np.sum(np.sum(np.square(clust_data - RepMat),1))) # In Square Root in Equation 11
        AllNValuesInCluster[clust_num] = NValuesInCluster
    WCD = (1/kval) * np.sum(InClust_WCD) # From Equation 11
    
    # Calculate between cluster distance (BCD)
    # Saitta et al. (2008): Intelligent Data Analysis 12 (2008) 529–548 529
    # "A comprehensive validity index for clustering"
    # Equation 10
    AllCentroid = np.mean(scaleData,axis=0) # Should be near 0 for each feature due to centering & scaling
    RepMat2 = repmat(AllCentroid,kval,1)
    BCD1 = np.sum(np.square(sk_centroids - RepMat2),axis=1)
    BCD2 = np.multiply(AllNValuesInCluster.transpose(),BCD1) # Element-by-Element Multiplication
    BCD3 = np.sum(BCD2)
    BCD = (1/(NValues*kval)) * BCD3
    
    SFScore = 1 - (1/(np.exp(np.exp(BCD - WCD))))
    
    SIScore = SI(scaleData,pred_classes_sk)
    
    # Return the important values
    return WCD, BCD, SFScore, SIScore