import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from GlassDataset_PyTorchGeometric import GlassyDataset
from torch_geometric.data import DataLoader
from NetClass_PyTorchGeometric import Net 


num_neighbors = 6
dataset = GlassyDataset('metadata/metadata.json', transform=T.NNGraph(num_neighbors))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

model.train()

data = iter(loader).next()
data.to(device)

optimizer.zero_grad()
output = model(data)

print(output)
print(output.size())
print(data.x.size())

exit()

loss = criterion(output, data.y)
# loss.backward()
# optimizer.step()
