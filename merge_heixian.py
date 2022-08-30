# coding=utf-8
import cv2

def merge_heixian(xs, ys, hs, ws, gray_values, x_dis=20, y_dis=150):
    # 先把x从小到大排序, 得到index顺序
    lens = len(xs)
    sorted_id = sorted(range(lens), key=lambda k: xs[k], reverse=False)
    xs_ = [xs[k] for k in sorted_id]
    ys_ = [ys[k] for k in sorted_id]
    hs_ = [hs[k] for k in sorted_id]
    ws_ = [ws[k] for k in sorted_id]
    gray_values_ = [gray_values[k] for k in sorted_id]
    
    # 从前向后遍历一次, 把需要合并的所有黑线段集合 
    all_merged = []
    visited = [1] * lens
    for i in range(lens-1):
        merged = [i]
        for j in range(i+1, lens):
            if (xs_[j] - xs_[i] <= x_dis) and (abs(ys_[j] - ys_[i]) <= y_dis) and visited[j]:
                merged.append(j)
                visited[j] = 0
                # print(merged)
        if visited[i]:
            all_merged.append(merged)
            visited[i] = 0
    
    # 处理最后一个hx
    near_ind = [ind for ind in range(lens-2, -1, -1) if ((xs_[ind]-xs_[lens-1]) <= x_dis) and (abs(ys_[ind] - ys_[lens-1]) <= y_dis)]
    if len(near_ind):
        near_index = [a for a in range(len(all_merged)) if near_ind[0] in all_merged[a]][0]
        all_merged[near_index].append(lens-1)
    else:
        all_merged.append([lens-1])
    
    merged_lens = len(all_merged)
    merged_x, merged_y, merged_w, merged_h, merged_gray_value = [0]*merged_lens,[0]*merged_lens,[0]*merged_lens,[0]*merged_lens,[0]*merged_lens
    for r, merged_index_list in enumerate(all_merged):
        if len(merged_index_list) == 1:
            merged_x[r] = xs_[merged_index_list[0]]
            merged_y[r] = ys_[merged_index_list[0]]
            merged_h[r] = hs_[merged_index_list[0]]
            merged_w[r] = ws_[merged_index_list[0]]  
        else:
            x0 = xs_[merged_index_list[0]] 
            y0 = min([ys_[p] for p in merged_index_list]) 
            x1 = max(xs_[p]+ws_[p] for p in merged_index_list)
            y1 = max(ys_[p]+hs_[p] for p in merged_index_list)
            merged_x[r] = x0
            merged_y[r] = y0
            merged_h[r] = y1-y0
            merged_w[r] = x1-x0

        # 合并的heixian gray_value, 赋值最黑value
        merged_gray_value[r] = min([gray_values_[p] for p in merged_index_list]) 

    return merged_x, merged_y, merged_w, merged_h, merged_gray_value
    
xs = [6670, 6665, 6787, 6258, 1216, 3982, 3964, 465, 466, 1174, 5540, 5525, 4364, 4364, 4366, 4375, 4372, 4364, 4363, 4357, 2734, 4357, 4355, 2617, 2530, 4373, 4377, 1734, 2658, 1736, 3440, 4393, 5773, 5765, 4359, 4357, 4364, 4355, 5336, 5525, 5303, 5525, 4354, 4370, 5323, 5303, 4408, 4379, 4298, 5780, 5789, 5767, 5679, 5666, 4359, 5605, 5590, 5563, 3819, 3828, 3808]
ys = [1896, 1917, 1979, 2283, 2714, 6378, 6882, 7715, 7816, 8171, 9793, 9915, 10314, 10467, 10522, 10672, 10749, 10780, 11746, 11839, 11841, 11933, 11958, 13723, 13736, 14767, 14836, 15255, 15292, 15299, 15928, 16088, 16179, 16312, 17123, 17138, 17300, 17336, 17372, 17374, 17476, 17515, 17649, 17710, 17727, 17764, 17911, 17935, 17972, 18148, 18179, 18221, 18420, 18607, 18623, 18744, 18775, 18789, 20431, 20472, 20534]
hs = [10, 300, 119, 91, 12, 18, 73, 96, 13, 16, 111, 10, 64, 28, 30, 16, 13, 70, 15, 86, 276, 12, 47, 95, 80, 50, 28, 32, 83, 55, 59, 169, 130, 10, 11, 27, 28, 109, 13, 126, 54, 91, 43, 25, 21, 20, 17, 34, 921, 18, 11, 147, 184, 59, 15, 18, 13, 283, 38, 21, 14]
ws = [4, 18, 24, 13, 3, 3, 10, 12, 9, 7, 12, 6, 11, 11, 13, 7, 5, 13, 5, 11, 13, 9, 9, 11, 11, 9, 5, 5, 7, 3, 13, 65, 63, 6, 7, 9, 11, 27, 7, 9, 13, 9, 14, 7, 6, 11, 11, 18, 117, 5, 9, 18, 61, 16, 4, 7, 6, 38, 9, 11, 6]
gray_values = [78.40833333333333, 78.03524691358025, 77.99929971988796, 49.981121442659905, 101.92592592592592, 107.8641975308642, 109.18538812785388, 108.39467592592592, 107.2905982905983, 107.79761904761905, 104.06581581581581, 98.45555555555555, 106.52698863636364, 105.7987012987013, 107.27179487179487, 106.74404761904762, 103.32820512820513, 106.44981684981686, 102.96, 105.68745595489781, 103.63089929394278, 103.98456790123457, 105.40583136327817, 101.1872408293461, 101.25227272727273, 107.79037037037037, 105.23333333333333, 46.395833333333336, 105.27825588066553, 39.19595959595959, 108.18035636679704, 100.40400546199363, 95.76190476190476, 39.74444444444445, 109.16017316017316, 107.2249657064472, 106.8409090909091, 108.97247706422019, 88.28571428571429, 102.78747795414462, 105.61111111111111, 101.81481481481481, 102.96677740863787, 100.97333333333333, 104.81746031746032, 100.91363636363636, 79.41711229946524, 82.91122004357298, 89.36796681422089, 77.95185185185186, 75.9057239057239, 78.97014361300076, 89.50985982418626, 80.9611581920904, 77.07777777777778, 76.10317460317461, 73.35897435897436, 88.4048106131052, 99.76608187134502, 85.3924963924964, 74.5]

merged_x, merged_y, merged_w, merged_h, merged_gray_value = merge_heixian(xs, ys, hs, ws, gray_values)
colored_predict = cv2.imread('./hx.jpg')
for ind, x in enumerate(merged_x):
    cv2.rectangle(colored_predict, (x, merged_y[ind]), (x + merged_w[ind], merged_y[ind]+merged_h[ind]), (0, 255, 0), 1)
    text = 'ind: {}, x:{}, y: {}, h: {}, w: {}, gray_value: {}'.format(ind, x, merged_y[ind], merged_h[ind], merged_w[ind], merged_gray_value[ind])
    cv2.putText(colored_predict, text, (x, merged_y[ind]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv2.imwrite('./merged_hx.jpg', colored_predict)

# colored_predict = cv2.imread('./hx.jpg')
# for ind, x in enumerate(xs):
#     cv2.rectangle(colored_predict, (x, ys[ind]), (x + ws[ind], ys[ind]+hs[ind]), (0, 255, 0), 1)
#     text = 'ind: {}, x:{}, y: {}, h: {}, w: {}, gray_value: {}'.format(ind, x, ys[ind], hs[ind], ws[ind], gray_values[ind])
#     cv2.putText(colored_predict, text, (x, ys[ind]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
# cv2.imwrite('./org_hx.jpg', colored_predict)
