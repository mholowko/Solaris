import numpy as np
import pandas as pd
from collections import defaultdict
import math
from sklearn.cluster import KMeans, AgglomerativeClustering

SORTED_GROUP = ['Consensus', 'BPS-C', 'BPS-NC', 'UNI', 'PPM', 'Bandit-0', 'Bandit-1']

def sort_kernel_matrix(df, feature_kernel, kmeans_based_on='label_distance'):
    group_dict = df.groupby('Group').groups
    groups = list(group_dict.keys())
    new_ordering = [] # idx ordering
    intersection = [i for i in SORTED_GROUP if i in set(groups)]
    print('groups: ', intersection)

    frr_seqs = np.asarray(df['RBS'])

    for group in intersection:
        df_group = df[df['Group'] == group]
        num_seqs = len(df_group)
        # num_clusters = int(num_seqs/5) + 1
        # print('Group: ', group)
        # print('Number of sequences: ', num_seqs)
        # print('number of clusters: ', num_clusters)
        
        idx = np.asarray(group_dict[group])
        # print('idx: ', idx)
        #print(label_distance[idx[0]: idx[-1], idx[0]: idx[-1]])
        
        
        if kmeans_based_on == 'label_distance': # kmeans based on label distances
            num_clusters = int(num_seqs/5) + 1
            # print('number of clusters: ', num_clusters)
            kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(np.asarray(df['AVERAGE'])[idx].reshape(len(idx),1))
            cluster_dict = defaultdict(list) # key: cluster id; value: idx list
            for i, cluster_id in enumerate(kmeans.labels_):
                cluster_dict[cluster_id].append(idx[i])
            # print('cluster dict: ', cluster_dict)
            # print('kmeans labels: ', kmeans.labels_)
            
        elif kmeans_based_on == 'seq_distance': # kmeans based on spectrum distances 
            num_clusters = int(num_seqs/8) + 1
            # print('number of clusters: ', num_clusters)
            #kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(phi_X[idx[0]: idx[-1] + 1, :])
            cluster_dict = defaultdict(list) # key: cluster id; value: idx list
            if len(idx) > 1:
                model = AgglomerativeClustering(n_clusters=num_clusters)
                model.fit(feature_kernel[idx])
                for i, cluster_id in enumerate(model.labels_):
                    cluster_dict[cluster_id].append(idx[i])
            # print('cluster dict: ', cluster_dict)
            # print('kmeans labels: ', model.labels_)
            else: # if the number of data points in one cluster is too small, just put them in one cluster
                cluster_dict[0] = idx

        # print('Sorting inside clusterings:')
        for key, value in cluster_dict.items():
            seq_list = []
            for i in value:
                seq_list.append(frr_seqs[i])
            # print('key: ', key)
            # print('seq list: ', seq_list)
            
            argsorted_seq_list = np.argsort(seq_list)
            # print('argsorted seq list: ', argsorted_seq_list)
            
            cluster_dict[key] = np.asarray(value)[np.asarray(argsorted_seq_list)]
        # print('sorted cluster dict: ', cluster_dict)
        
        # print('Sorting clusterings:')
        
        
        if kmeans_based_on == 'label_distance':
            # print('kmeans cluster center: ', kmeans.cluster_centers_)
            argsorted_cluster_ids = np.argsort(kmeans.cluster_centers_.reshape(num_clusters,))[::-1]
        else:
            # TODO: check
            argsorted_cluster_ids = range(num_clusters)
            
        # print('argsort kmeans cluster center: ', argsorted_cluster_ids)
        for cluster_id in argsorted_cluster_ids:
            for i in cluster_dict[cluster_id]:   
                new_ordering.append(i)
                
        # print('new ordering: ', new_ordering)
        # print()              

    return feature_kernel[:, new_ordering][new_ordering,:], new_ordering                                    
