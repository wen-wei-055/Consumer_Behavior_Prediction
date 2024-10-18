import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import models
import loader
device = torch.device(1 if torch.cuda.is_available() else "cpu") 

parser = argparse.ArgumentParser()
parser.add_argument('--weight_path', type=str, required=True)
args = parser.parse_args()

# Construct Model
fullmodel = models.FullModel(device).to(device)
fullmodel.load_state_dict(torch.load(args.weight_path)['model_weights'])

# Dataset
test_datas = loader.read_dataset('Test')

# Evaluate
times = np.arange(1, 100, 1) # 1~100個行為
final_true_array = np.zeros((len(test_datas), len(times)))
final_pred_array = np.zeros((len(test_datas), len(times)))
for test_data_i, test_data in enumerate(tqdm(test_datas)):
    cutout_generators = loader.CutoutGenerator([test_data], device, times)
    test_dataloader = DataLoader(cutout_generators, batch_size=1, shuffle=False)

    # 每個行為事件序列個別存下
    pred_list = []  # (行為數=20,)
    true_list = []  # (行為數=20,)
    fullmodel.eval()
    with torch.no_grad(): 
        for i, (member_vectors, member_age, behavior_vector, label) in enumerate(test_dataloader):
            # print('behavior_vector',behavior_vector[behavior_vector==0])
            pred = fullmodel(member_vectors, member_age, behavior_vector)
            label
            
            ############ conf matrix ############
            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.detach().cpu().numpy().tolist()
            true_list += label.detach().cpu().numpy().tolist()
            
    final_pred_array[test_data_i] = pred_list
    final_true_array[test_data_i] = true_list
    
    # if test_data_i == 1000:
    #     break

np.save('evaluate_pred', final_pred_array)
np.save('evaluate_true', final_true_array)