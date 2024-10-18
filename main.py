import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import models
import shutil
import loader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
device = torch.device(0 if torch.cuda.is_available() else "cpu") 
np.random.seed(42)
torch.manual_seed(42) #CPU seed
torch.cuda.manual_seed(42) #GPU seed

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
args = parser.parse_args()

# Store training files
for doc_name in ['main.py','evaluate.py','models.py','loader.py']:
    shutil.copyfile(doc_name, f'{args.path}/{doc_name}')

# DataLoader
print('Loading Dataset...')
train_dataset = loader.Dataset(loader.read_dataset('Train'), device)
dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = loader.Dataset(loader.read_dataset('Test'), device)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
# Construct Model
class_weights = [1, 10]
class_weights = torch.FloatTensor(class_weights).to(device)
    
fullmodel = models.FullModel(device).to(device)
loss = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(fullmodel.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2)


# Train / Test
print(f'Total Training Set: {len(train_dataset)}')
print(f'Total Testing Set: {len(test_dataset)}')
epochs = 100
for epoch in range(epochs):
    ############################ Train ############################
    total_train_loss = 0.0
    pred_list = []
    true_list = []
    fullmodel.train()
    for i, (member_vectors, member_age, behavior_vector, label) in enumerate(tqdm(dataloader)):
        # print(f"Member Vectors: {member_vectors}")
        # print(f"Behavior member_age: {member_age}")
        # print(f"Behavior Vector: {behavior_vector}")
        # print(f"Labels: {labels}")
        pred = fullmodel(member_vectors, member_age, behavior_vector)
        
        ############ loss ############
        train_loss = loss(pred, label)
        total_train_loss = train_loss.item() + total_train_loss
        
        train_loss.backward()
        optimizer.step()          # Update Weights     
        optimizer.zero_grad() 
        
        ############ conf matrix ############
        _, predicted = torch.max(pred.data, 1)
        pred_list += predicted.detach().cpu().numpy().tolist()
        true_list += label.detach().cpu().numpy().tolist()
        # if i == 20:
        #     break
    
    cm = confusion_matrix(true_list, pred_list)
    precision = precision_score(true_list, pred_list, average=None)
    recall = recall_score(true_list, pred_list, average=None)
    f1 = f1_score(true_list, pred_list, average=None)
    with open(f'{args.path}/train_conf_matrix.txt', 'a', encoding='utf-8') as f:
        f.write(f'epoch: {epoch}\n{str(cm)}\nloss: {total_train_loss / len(dataloader)},  pre:{precision},  rec:{recall},  F1:{f1},  lr:{scheduler.optimizer.param_groups[0]["lr"]}\n\n\n')


    ############################ Valid ############################
    total_val_loss = 0.0
    pred_list = []
    true_list = []
    fullmodel.eval()
    for i, (member_vectors, member_age, behavior_vector, label) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            pred = fullmodel(member_vectors, member_age, behavior_vector)
        
            ############ loss ############
            val_loss = loss(pred, label)
            total_val_loss = val_loss.item() + total_val_loss
            
            ############ conf matrix ############
            _, predicted = torch.max(pred.data, 1)
            pred_list += predicted.detach().cpu().numpy().tolist()
            true_list += label.detach().cpu().numpy().tolist()
        # if i == 20:
        #     break
        
    scheduler.step(total_val_loss/len(test_dataloader))
            
    cm = confusion_matrix(true_list, pred_list)
    precision = precision_score(true_list, pred_list, average=None)
    recall = recall_score(true_list, pred_list, average=None)
    f1 = f1_score(true_list, pred_list, average=None)
    with open(f'{args.path}/val_conf_matrix.txt', 'a', encoding='utf-8') as f:
        f.write(f'epoch: {epoch}\n{str(cm)}\nloss: {total_val_loss / len(test_dataloader)},  pre:{precision},  rec:{recall},  F1:{f1},  lr:{scheduler.optimizer.param_groups[0]["lr"]}\n\n\n')
        
    torch.save({'model_weights' : fullmodel.state_dict()}, f'{args.path}/checkpoint{epoch}.pth')