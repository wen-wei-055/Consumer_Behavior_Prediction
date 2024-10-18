import numpy as np
import argparse
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def accumulate_ones(sequence, one_dim=False):
    # 計算累積和，只針對 1
    if one_dim == False:
        cumulative_sum = np.cumsum(sequence, axis=1)  # 先計算整體的累和
    if one_dim == True:
        cumulative_sum = np.cumsum(sequence)  # 先計算整體的累和
    sequence[sequence == 1] = cumulative_sum[sequence == 1]  # 只更新數列1的位置
    return sequence

def create_masked_matrix(array):
    # 獲取原始數組的長度
    n = len(array)
    # 創建一個二維布爾矩陣，其中第 i 行有 i 個 True
    mask = np.tri(n, dtype=bool, k=0)
    # 利用廣播將原始數組中的元素放到新的二維矩陣中
    matrix = np.where(mask, array[:, None], 0)
    return matrix


parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, required=True)   #放 evaluate_true 和 evaluate_pred 的資料夾
parser.add_argument('--pred_count', type=int, required=True)

parser.add_argument('--normal_pred', action='store_true')     #擇一
parser.add_argument('--regression_pred', action='store_true') #擇一

args = parser.parse_args()

if args.pred_count <= 0:
    raise ValueError('The value of pred_count needs to be greater than 0')

evaluate_label = np.load(f'{args.weight_path}/evaluate_true.npy') 
evaluate_pred = np.load(f'{args.weight_path}/evaluate_pred.npy')

# TP: label=1, pred=1
# FP: label=0, pred=1
# TN: label=0, pred=0
# FN: label=1, pred=0

# 儲存格式
# TP: [第幾秒預測, ]
# FP: [第幾秒誤報, ]
# TN: 總數量
# FN: 總數量

if args.normal_pred:
    TP = []
    FP = []
    TN = 0
    FN = 0
    # 讓pred變成只在count=x的時候=True, 其餘時間=False
    evaluate_pred = accumulate_ones(evaluate_pred)
    pred_count_bool = evaluate_pred==args.pred_count

    LabelPos = np.any(evaluate_label!=0, 1)
    LabelNeg = ~np.any(evaluate_label!=0, 1)
    PredPos = np.any(pred_count_bool, 1)
    PredNeg = ~np.any(pred_count_bool, 1)

    # 四指標id位置
    TP_idx = np.where(np.logical_and(LabelPos, PredPos))[0]
    FP_idx = np.where(np.logical_and(LabelNeg, PredPos))[0]
    TN_idx = np.where(np.logical_and(LabelNeg, PredNeg))[0]
    FN_idx = np.where(np.logical_and(LabelPos, PredNeg))[0]

    # 計算最終結果
    TN = len(TN_idx)
    FN = len(FN_idx)

    for i in TP_idx:
        TP.append(np.argmax(pred_count_bool[i]))\

    for i in FP_idx:
        FP.append(np.argmax(pred_count_bool[i]))
        
    with open(f'count{args.pred_count}_analysis.txt','w') as f:
        f.write(f'{args.pred_count} 次Pred=True決定要預測為會購買\n')
        f.write(f'TP: {len(TP)} | {TP}\n')
        f.write(f'FP: {len(FP)} | {FP}\n')
        f.write(f'TN: {TN}\n')
        f.write(f'FN: {FN}\n')
        
        

if args.regression_pred:  #(行為數, 100, 4)
    #抓出第n個點發報, 把發報點以後的行為序列補1
    final_output = np.zeros((evaluate_pred.shape[0],evaluate_pred.shape[1],4))
    for i, evaluate_pred_i in enumerate(tqdm(evaluate_pred)):
        # print('evaluate_pred_i',evaluate_pred_i)
        evaluate_pred_i = accumulate_ones(evaluate_pred_i, one_dim=True)
        # print('evaluate_pred_i',evaluate_pred_i)
        # 以上皆ok
        
        pred_count_idx = np.where(evaluate_pred_i==args.pred_count)[0]
        evaluate_pred_i = evaluate_pred_i==args.pred_count #變成只有發報時間點[0,0,1,0,0,0]
        # print('evaluate_pred_i',evaluate_pred_i)
        if len(pred_count_idx) > 0:
            #把發報點以後的行為序列補1
            evaluate_pred_i[pred_count_idx[0]:] = 1
        # evaluate_pred_i [1,0,1,0,0,0] -> [0,0,1,1,1,1]
        
        evaluate_pred_i = create_masked_matrix(evaluate_pred_i)  # 依時間段遮罩(100,) -> (100,100)
        # 看斜角就好
        evaluate_pred_i = np.diag(evaluate_pred_i)  # (100,100) > (100,)
        
        LabelPos = evaluate_label[i]!=0
        LabelNeg = evaluate_label[i]==0
        PredPos = evaluate_pred_i !=0
        PredNeg = evaluate_pred_i == 0
        # 四指標id位置
        # TP: label=1, pred=1
        # FP: label=0, pred=1
        # TN: label=0, pred=0
        # FN: label=1, pred=0
        TP_idx = np.expand_dims(np.logical_and(LabelPos, PredPos),-1)  #(100,100)
        FP_idx = np.expand_dims(np.logical_and(LabelNeg, PredPos),-1)  #(100,100)
        TN_idx = np.expand_dims(np.logical_and(LabelNeg, PredNeg),-1)  #(100,100)
        FN_idx = np.expand_dims(np.logical_and(LabelPos, PredNeg),-1)  #(100,100)  
        tmp = np.concatenate([TP_idx, FP_idx, TN_idx, FN_idx], -1)
        final_output[i] = tmp
        # print(final_output[i])
        # if i == 1000:
        #     break
        
    f = open(f'regression_analysis.txt','w')   
    print('製作 時間軸混淆矩陣......')
    tmp_path = f'{args.weight_path}/confusion_matrix_series/'
    for i, length_index in tqdm(enumerate(range(final_output.shape[1]))):
        tp = final_output[:, length_index, 0]
        fp = final_output[:, length_index, 1]
        tn = final_output[:, length_index, 2]
        fn = final_output[:, length_index, 3]

        y_true = tp + fn  
        y_pred = tp + fp  

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        f.write(f"Length {length_index}:")
        f.write(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}\nConfusion Matrix:\n{conf_matrix}\n')
        f.write('\n')    
                
        # 行规范化
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        norm_conf_matrix = conf_matrix / row_sums

        # 绘制混淆矩阵
        plt.figure(figsize=(10.5,10))
        plt.imshow(norm_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'behavior number: {i}', fontsize=30, pad=10)
        # plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true)))  # Assume labels are from 0 to n_classes-1
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        
        plt.tight_layout()
        plt.xticks(tick_marks, ['not buy', 'buy'], fontsize=25)  # 设置x轴刻度标签
        plt.yticks(tick_marks, ['not buy', 'buy'], fontsize=25)     
        plt.ylabel('True', fontsize=25)
        plt.xlabel('Pred', fontsize=25)
        plt.tick_params(axis='x', which='major', pad=10)  # 调整 x 轴刻度标签的距离
        plt.tick_params(axis='y', which='major', pad=15)  # 调整 y 轴刻度标签的距离

        # 在矩阵中添加数值标注
        thresh = norm_conf_matrix.max() / 2
        for ii in range(conf_matrix.shape[0]):
            for jj in range(conf_matrix.shape[1]):
                plt.text(jj, ii, format(conf_matrix[ii, jj], 'd'),
                        horizontalalignment='center',
                        color='white' if norm_conf_matrix[ii, jj] > thresh else 'black', 
                        fontsize=25)

        # 调整子图的布局参数
        plt.subplots_adjust(left=0.2, bottom=0.1)  # 增大左边距，使整个图表向右移动
        
        # 保存图像到文件
        if not os.path.isdir(tmp_path): os.makedirs(tmp_path)
        plt.savefig(f'{tmp_path}{str.format("{:02}", i)}_confusion_matrix.png')
        plt.close()  # 关闭图形，防止显示出来
    
    print('製作 影片......')
    image_files = sorted([img.split('.')[0] for img in os.listdir(tmp_path) if img.endswith('.png')])
    image_files = sorted(image_files)
    
    # 設定影片參數
    # 設定新的帧速率以延長影片持續時間
    frame_rate = 3  # 每秒1帧，持續5秒，总共5帧
    # 創建一個VideoWriter對象以將圖像合併為影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    image = cv2.imread(tmp_path+image_files[0]+'.png')
    height, width, layers = image.shape
    
    # 創建一個VideoWriter對象以將圖像合併為影片
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{args.weight_path}_prediction.mp4', fourcc, frame_rate, (width, height))

    for i, image_file in enumerate(image_files):
        image_file = tmp_path + image_file+'.png'
        image = cv2.imread(image_file)
        out.write(image)
    
    # 釋放VideoWriter對象並關閉影片文件
    out.release()
    cv2.destroyAllWindows()
