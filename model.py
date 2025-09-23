import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transforms = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),])
train_dataset = datasets.Food101(root='./data', split='train', transform = transforms, download = False)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1,2,0))
plt.title(labels[0].item())
plt.show()