import numpy as np
import torch
import json


def read_dataset(type):
    if type == 'Train':
        dataset0 = json.load(open('../dataset/session02_202309_new.json','r'))
        dataset1 = json.load(open('../dataset/session02_202310_new.json','r'))
        dataset2 = json.load(open('../dataset/session02_202311_new.json','r'))
        dataset3 = json.load(open('../dataset/session02_202312_new.json','r'))
        datasets = dataset0 + dataset1 + dataset2 + dataset3
    
    elif type == 'Test':
        dataset4 = json.load(open('../dataset/session02_202401_new.json','r'))
        dataset5 = json.load(open('../dataset/session02_202402_new.json','r'))
        datasets = dataset4 + dataset5 
    
    else: raise ValueError('Unknown Dataset Type')
    
    return datasets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device, cutout=None):
        self.device = device
        self.cutout = cutout
        self.behavior_len = 100
        self.dataset = dataset
        # self.member_categories_per_dimension = [6,3,2,2,2,2]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):  #i=時間點
        # 分割後8: 會員資料
        # [RegisterSourceTypeDef  Gender  IsAppInstalled  IsEnableEmail  IsEnablePushNotification  IsEnableShortMessage  MemberCardLevel  Age]
        # RegisterSourceTypeDef: 5
        # Gender: 3
        # IsAppInstalled  IsEnableEmail  IsEnablePushNotification  IsEnableShortMessage: 2
        # MemberCardLevel: 10
        # Age: continuous
        
        # 分割後8以前: 行為資料
        
        label = self.dataset[i][1]
        behavior_dataset = self.dataset[i][0][:-8]
        # 行為資料: 長度=self.behavior_len, 其餘補0
        behavior_vectors = np.zeros((self.behavior_len))
        behavior_vectors[:len(behavior_dataset)] = behavior_dataset[:self.behavior_len]
        # 行為資料隨機補0, 以模擬真實情境
        # cutout = np.random.randint(2, min(len(behavior_dataset),self.behavior_len))
        if self.cutout:
            cutout = self.cutout
        else:
            cutout = np.random.randint(0, max(1, min(len(behavior_dataset),self.behavior_len)))
        behavior_vectors[cutout:] = 0

        # # 會員資料類別項one-hot / 會員資料年紀除以100
        # member_vectors = []
        # for idx, category in enumerate(member_dataset):
        #     category_count = self.member_categories_per_dimension[idx]
        #     onehot_vector = np.eye(category_count)[category]
        #     member_vectors.append(onehot_vector)
            
        # member_vectors.append(np.eye(10)[int(member_CardLevel)])
        # member_vectors = np.concatenate(member_vectors).tolist()
        # member_vectors += [member_age]
        member_vectors = self.dataset[i][0][-8:-2] + [int(self.dataset[i][0][-2] / 10)]
        # member_CardLevel = self.dataset[i][0][-2] / 10
        # print('member_dataset',member_dataset)
        member_age = self.dataset[i][0][-1] / 100
        
        member_vectors = torch.tensor(member_vectors).to(self.device)
        member_age = torch.tensor(member_age).to(self.device)
        behavior_vectors = torch.from_numpy(behavior_vectors.astype(int)).to(self.device)
        label = torch.tensor(label).to(self.device)
        
        return member_vectors, member_age, behavior_vectors, label
    
    
# def CutoutGenerator(dataset, device, times):
#     generator_list = []
#     for time in times:
#         generator = Dataset(dataset, device, time)
#         generator_list.append(generator)
#     return generator_list
    
class CutoutGenerator(torch.utils.data.Dataset):
    def __init__(self, dataset, device, times):
        self.times = times
        self.generator = Dataset(dataset, device)
        self.indexes = []
            
    def __len__(self):
        return len(self.generator) * len(self.times)

    def __getitem__(self, index):
        time = self.times[index]
        self.generator.cutout = time
        return self.generator[0]   # 0是因為evaluate寫法下，資料集只有一筆
