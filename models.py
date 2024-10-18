import torch
import torch.nn.functional as F
import torch.nn as nn



class MultiEmbeddings(nn.Module):
    def __init__(self, device, num_categories_per_field, embedding_dim):
        super(MultiEmbeddings, self).__init__()
        num_categories_per_field = num_categories_per_field
        embedding_dim = embedding_dim
        self.embeddings = nn.ModuleList([nn.Embedding(num_categories, embedding_dim) for num_categories in num_categories_per_field])
    
    def forward(self, x):
        embedded_fields = [embedding(x[:, i]).unsqueeze(1) for i, embedding in enumerate(self.embeddings)]
        return torch.cat(embedded_fields, dim=1)


class Embeddings(nn.Module):
    def __init__(self, device, category, embedding_dim):
        super(Embeddings, self).__init__()
        embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(category, embedding_dim)
    
    def forward(self, x):
        x = self.embeddings(x)
        return x


class MemberEncoder(nn.Module):
    def __init__(self, device):
        super(MemberEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3))  # No padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))  # No padding
        self.pool = nn.MaxPool2d(2, 2)  # Pooling to reduce size
        self.relu = nn.ReLU()
        # For 1D Conv, adjust number of input channels according to the output of last 2D conv layer
        self.conv1d = nn.Conv1d(32, 64, kernel_size=3)  # No padding
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
    def forward(self, x):
        # x shape: [batch_size, 1, 7, 10]
        x = self.relu(self.conv1(x.unsqueeze(1)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten for transition to 1D conv layers
        x = x.view(x.size(0), 32, 3)  # Adjust shape for 1D Conv
        x = self.conv1d(x)
        x = self.global_pool(x) #(batch, 64, 1)
        x = x.view(x.size(0), -1) #(batch, 64)
        return x
    
    
class BehaviorEncoder(nn.Module):
    def __init__(self, device):
        super(BehaviorEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))  # No padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 1), stride=(1, 1))  # No padding
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 1))  # Pooling only in one dimension

        self.conv1d_1 = nn.Conv1d(32, 45, kernel_size=3, stride=1)  # No padding
        self.conv1d_2 = nn.Conv1d(45, 64, kernel_size=3, stride=1)  # No padding
        
        self.Linear = nn.Linear(44, 1)
        # self.global_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = x.unsqueeze(1)  # torch.Size([2, 1, 100, 3])
        x = self.relu(self.conv1(x))  # torch.Size([2, 16, 98, 1])
        x = self.relu(self.conv2(x))  # torch.Size([2, 32, 96, 1])
        x = self.pool(x)  # torch.Size([2, 32, 48, 1])
        x = x.squeeze(3)  # torch.Size([2, 32, 48])
        
        x = self.relu(self.conv1d_1(x))  # torch.Size([2, 45, 46])
        x = self.relu(self.conv1d_2(x))  # torch.Size([2, 64, 44])
        x = self.Linear(x).squeeze(-1)  # torch.Size([2, 64])        
        return x


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(64, 50)  # 第一层线性变换，从138降到50
        self.fc2 = nn.Linear(50, 20)   # 第二层线性变换，从50降到20
        self.fc3 = nn.Linear(20, 2)    # 第三层线性变换，从20降到2

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用ReLU激活函数
        x = F.relu(self.fc2(x))  # 使用ReLU激活函数
        x = self.fc3(x)
        return x



class FullModel(nn.Module):
    def __init__(self, device='cpu'):
        super(FullModel, self).__init__()
        self.device = device
        
        self.MemberEmbeddings = MultiEmbeddings(device, [6,3,2,2,2,2,10], 10)
        self.BehaviorEmbeddings = Embeddings(device, 17, 3)
        self.MemberEncoder = MemberEncoder(device)
        self.BehaviorEncoder = BehaviorEncoder(device)
        self.AgeLinear = nn.Linear(1, 10, bias=True)
        self.Decoder = Decoder(device)

    def forward(self, member_x, member_age, behavior_x):   #(batch,8) / (batch,) / (batch,100)
        member_x = self.MemberEmbeddings(member_x)
        behavior_x = self.BehaviorEmbeddings(behavior_x)

        member_x = self.MemberEncoder(member_x)
        behavior_x = self.BehaviorEncoder(behavior_x)

        member_age = member_age.unsqueeze(-1)
        member_age = self.AgeLinear(member_age) #member_age torch.Size([2, 10])

        x = behavior_x #torch.Size([2, 138])
        x = self.Decoder(x)
        return x

