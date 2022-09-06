# coding=utf-8
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN



def dbscan_fushidian(xs, ys, ws, hs, scores, areas, min_samples=4, R=50, trian_eps=False):

    '''
    min_samples 前端可调参数, 聚类最小样本数
    trian_eps 是否开启最佳eps搜索, 词过程较耗时
    '''
    
    
    # X: dl模型判断检出的 fushidian的box-center-list
    X = [[xs[ind]+ws[ind]//2, ys[ind]+hs[ind]//2] for ind in range(lens)]
    lens = len(X)
    X= [[p[0]/8000, p[1]/20000] for p in X]
    
    if trian_eps:
        cluster_num = []
        for eps_ in np.arange(0.0001, 0.2, 0.00001):
            dbscan = DBSCAN(eps=eps_, min_samples=min_samples).fit(X)
            cluster_num.append(len(list(set(dbscan.labels_))))

        ycy_eps = cluster_num.index(max(cluster_num))*0.00001+0.0001

    else:
        ycy_eps = 0.0753
    

    dbscan = DBSCAN(eps=ycy_eps, min_samples=min_samples).fit(X)
    y_pre = dbscan.labels_
    # -1是噪声类, 认为不是fushidian, 直接舍弃
    cluster_labels = [a for a in list(set(y_pre)) if a != -1]
    remain_inds = [r_ind for r_ind in range(lens) if y_pre[r_ind] != -1]
    print('remain_inds: ', len(remain_inds))

    cluster_center_xy = [[] for a in range(len(cluster_labels))]
    xs_, ys_ = [], []
    scores_, areas_ = [], []
    ws_, hs_ = [], []

    for cluster_index, cluster in enumerate(cluster_labels):
        cluster_x = [X[a][0] for a in range(lens) if y_pre[a] == cluster]
        cluster_y = [X[a][1] for a in range(lens) if y_pre[a] == cluster]
        cluster_center_xy[cluster_index] = [int(np.mean(cluster_x)*8000), int(np.mean(cluster_y)*20000)]
        if not R:
            # 不做聚类中心距离约束, 保留全部聚类到的点
            xs_, ys_ = [xs[k] for k in remain_inds], [ys[k] for k in remain_inds]
            ws_, hs_ = [ws[k] for k in remain_inds], [hs[k] for k in remain_inds]
            scores_, areas_ = [scores[k] for k in remain_inds], [areas[k] for k in remain_inds]
        else:
            center_x_min, center_x_max = max(0,int(np.mean(cluster_x)*8000)-R), min(int(np.mean(cluster_x)*8000)+R, 8192)
            center_y_min, center_y_max = max(0, int(np.mean(cluster_y)*20000)-R), min(int(np.mean(cluster_y)*20000)+R, 22000)
            print("x_center: ", center_x_min, center_x_max)
            print("y_center: ", center_y_min, center_y_max)
            
            xs_ = [xs[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]
            ys_ = [ys[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]
            scores_ = [scores[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]
            areas_ = [areas[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]
            ws_ = [ws[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]
            hs_ = [hs[ind] for ind in remain_inds if ((center_x_min<=xs[ind]<=center_x_max) and (center_y_min<=ys[ind]<=center_y_max))]


    return cluster_center_xy, xs_, ys_, ws_, hs_, scores_, areas_