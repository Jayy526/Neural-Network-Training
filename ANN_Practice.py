import  pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader,Dataset


df = pd.read_csv('Breast_cacner.csv')
df.drop(columns=['id','Unnamed: 32'],inplace=True)
X_train,X_test,y_train,y_test = train_test_split(df.iloc[:,1:],df.iloc[:,0],test_size=0.2,random_state=11)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train).long()
y_test_tensor = torch.from_numpy(y_test).long()

class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.features)
    def __getitem__(self, index):
        return self.features[index],self.labels[index]

train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class MyNN(nn.Module):
    def __init__(self,num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,2),

        )
    def forward(self,x):
        return self.model(x)
epochs = 50
learning_rate = 0.001
model = MyNN(X_train_tensor.shape[1])
optimizer = optim.Adam(model.parameters(),lr= learning_rate)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(epochs):
    total_epoch_loss = 0
    for batch_features,batch_labels in train_dataloader:
        outputs = model(batch_features)
        loss = loss_fn(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()
    avg_loss = total_epoch_loss/len(train_dataloader)
    print(f'Epoch: {epoch + 1}, Loss: {avg_loss}')


model.eval()
total = 0
correct = 0

with torch.no_grad():

  for batch_features, batch_labels in test_dataloader:

    outputs = model(batch_features)

    _, predicted = torch.max(outputs, 1)

    total = total + batch_labels.shape[0]

    correct = correct + (predicted == batch_labels).sum().item()

print(correct/total)