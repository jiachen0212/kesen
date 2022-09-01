# coding=utf-8
import cv2


def merged_fun(cur_ind, xs, ys, visited, hs, x_dis_thres, y_dis_thres):
    merged_list = []
    lens = len(xs)
    for j in range(cur_ind+1, lens):
        if visited[j] and abs(xs[cur_ind]-xs[j]) <= x_dis_thres:
            # 先计算两个hx的相对上下位置, 再取上的下边界, 和下的上边界, 计算计算距离即可
            if ys[cur_ind] > ys[j]:
                y1, y2 = hs[j]+ys[j], ys[cur_ind]
            else:
                y1, y2 = hs[cur_ind]+ys[cur_ind], ys[j]
            if abs(y1-y2) <= y_dis_thres:
                merged_list.append(j)
                visited[j] = 0

    return merged_list


def dfs(xs, ys, hs, x_dis_thres, y_dis_thres):
    all_merged = []
    lens = len(xs)
    visited = [1]*lens
    
    for cur_index in range(lens):
        if visited[cur_index]:
            stack  = [cur_index]
            merged_hx = [cur_index]
            visited[cur_index] = 0
            while stack:
                # 弹出最后一个x, 则x相关的merged_list要加入stack
                cur_ = stack.pop()
                merged_list = merged_fun(cur_, xs, ys, visited, hs, x_dis_thres, y_dis_thres)
                stack.extend(merged_list)
                merged_hx.extend(merged_list)
            for ind in merged_hx:
                visited[ind] = 0
            all_merged.append(merged_hx)
        
    return all_merged


def merged_box(xs, ys,  hs, gray_values, x_dis_thres=None, y_dis_thres=None):
    all_merged = dfs(xs, ys,  hs, x_dis_thres, y_dis_thres)
    merged_lens = len(all_merged)
    merged_box_grayvalue_score_area = [[0]*merged_lens for i in range(2)]
    for r, merged_index_list in enumerate(all_merged):
        if len(merged_index_list) == 1:
            x0, y0 = xs[merged_index_list[0]], ys[merged_index_list[0]]
            x1 = x0 + ws[merged_index_list[0]]
            y1 = y0 + hs[merged_index_list[0]]  
        else:
            x0 = min([xs[p] for p in merged_index_list])  
            y0 = min([ys[p] for p in merged_index_list]) 
            x1 = max([xs[p]+ws[p] for p in merged_index_list])
            y1 = max([ys[p]+hs[p] for p in merged_index_list])
        merged_box_grayvalue_score_area[0][r] = [x0, y0, x1, y1]
                
        # 合并的heixian gray_value, 赋值最黑value
        merged_box_grayvalue_score_area[1][r] = min([gray_values[p] for p in merged_index_list]) 

    return merged_box_grayvalue_score_area



xs = [6670, 6665, 6787, 6258, 1216, 3982, 3964, 465, 466, 1174, 5540, 5525, 4364, 4364, 4366, 4375, 4372, 4364, 4363, 4357, 2734, 4357, 4355, 2617, 2530, 4373, 4377, 1734, 2658, 1736, 3440, 4393, 5773, 5765, 4359, 4357, 4364, 4355, 5336, 5525, 5303, 5525, 4354, 4370, 5323, 5303, 4408, 4379, 4298, 5780, 5789, 5767, 5679, 5666, 4359, 5605, 5590, 5563, 3819, 3828, 3808]
ys = [1896, 1917, 1979, 2283, 2714, 6378, 6882, 7715, 7816, 8171, 9793, 9915, 10314, 10467, 10522, 10672, 10749, 10780, 11746, 11839, 11841, 11933, 11958, 13723, 13736, 14767, 14836, 15255, 15292, 15299, 15928, 16088, 16179, 16312, 17123, 17138, 17300, 17336, 17372, 17374, 17476, 17515, 17649, 17710, 17727, 17764, 17911, 17935, 17972, 18148, 18179, 18221, 18420, 18607, 18623, 18744, 18775, 18789, 20431, 20472, 20534]
hs = [10, 300, 119, 91, 12, 18, 73, 96, 13, 16, 111, 10, 64, 28, 30, 16, 13, 70, 15, 86, 276, 12, 47, 95, 80, 50, 28, 32, 83, 55, 59, 169, 130, 10, 11, 27, 28, 109, 13, 126, 54, 91, 43, 25, 21, 20, 17, 34, 921, 18, 11, 147, 184, 59, 15, 18, 13, 283, 38, 21, 14]
ws = [4, 18, 24, 13, 3, 3, 10, 12, 9, 7, 12, 6, 11, 11, 13, 7, 5, 13, 5, 11, 13, 9, 9, 11, 11, 9, 5, 5, 7, 3, 13, 65, 63, 6, 7, 9, 11, 27, 7, 9, 13, 9, 14, 7, 6, 11, 11, 18, 117, 5, 9, 18, 61, 16, 4, 7, 6, 38, 9, 11, 6]
gray_values = [78.40833333333333, 78.03524691358025, 77.99929971988796, 49.981121442659905, 101.92592592592592, 107.8641975308642, 109.18538812785388, 108.39467592592592, 107.2905982905983, 107.79761904761905, 104.06581581581581, 98.45555555555555, 106.52698863636364, 105.7987012987013, 107.27179487179487, 106.74404761904762, 103.32820512820513, 106.44981684981686, 102.96, 105.68745595489781, 103.63089929394278, 103.98456790123457, 105.40583136327817, 101.1872408293461, 101.25227272727273, 107.79037037037037, 105.23333333333333, 46.395833333333336, 105.27825588066553, 39.19595959595959, 108.18035636679704, 100.40400546199363, 95.76190476190476, 39.74444444444445, 109.16017316017316, 107.2249657064472, 106.8409090909091, 108.97247706422019, 88.28571428571429, 102.78747795414462, 105.61111111111111, 101.81481481481481, 102.96677740863787, 100.97333333333333, 104.81746031746032, 100.91363636363636, 79.41711229946524, 82.91122004357298, 89.36796681422089, 77.95185185185186, 75.9057239057239, 78.97014361300076, 89.50985982418626, 80.9611581920904, 77.07777777777778, 76.10317460317461, 73.35897435897436, 88.4048106131052, 99.76608187134502, 85.3924963924964, 74.5]


merged_box_grayvalue_score_area = merged_box(xs, ys,  hs, gray_values, x_dis_thres=20, y_dis_thres=150)
boxs = merged_box_grayvalue_score_area[0]
grays =  merged_box_grayvalue_score_area[1]
colored_predict = cv2.imread('./hx.jpg')
for ind, bbox in enumerate(boxs):
    x0,y0,x1,y1 = bbox[:4]
    cv2.rectangle(colored_predict, (x0, y0), (x1, y1), (0, 255, 0), 1)
    text = 'ind: {}, x:{}, y: {}, h: {}, w: {}, gray_value: {}'.format(ind, x0,y0,x1-x0,y1-y0,grays[ind])
    cv2.putText(colored_predict, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
cv2.imwrite('./merged_hx.jpg', colored_predict)




logo_mask = cv2.imread('./test_apple_mask.jpg')
org_image = cv2.imread(r'D:\work\project\beijing\Smartmore\2022\DL\kesen\codes\sdk_test\suidao\test_dir\A180_KSA0000000879003_Snow_Station4_Linear_Tunnel_2_2022_08_29_16_01_19_196_RC_N_Ori.bmp')
show_merged_mask = cv2.addWeighted(logo_mask, 0.1, org_image, 0.9, 10)    
cv2.imwrite('./show_merged_mask.jpg', show_merged_mask)   