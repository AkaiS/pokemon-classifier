import torch
from torch.utils.data import (Dataset,
                              DataLoader,
                              random_split)
import os
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import (Compose,
                                       Resize,
                                       RandomHorizontalFlip,
                                       ToDtype,
                                       Normalize)
from model import ResidualConvolutionalNN

# hyperparameters
lr = .1
num_epochs = 101
device = 'cuda' if torch.cuda.is_available() else 'cpu'
momentum = .9
weight_decay = .0001
batch_size = 64
img_dir_name = 'PokemonData'
patience = 5

# data location and label numerization
img_dir = os.path.abspath(img_dir_name)
img_labels = sorted(os.listdir(img_dir))
stoi = {l: i for i, l in enumerate(img_labels)}
itos = {i: l for i, l in enumerate(img_labels)}

# build class for pokemon image dataset
class PokemonImageDataset(Dataset):

  def __init__(self, img_dir, img_labels, labels_to_idx, transform=None):
    self.img_dir = img_dir
    self.transform = transform
    self.img_labels = img_labels
    self.labels_to_idx = labels_to_idx
    self.file_list = self._build_file_list()

  def __len__(self):
    return len(self.file_list)
  
  def __getitem__(self, idx):
    img_path, label = self.file_list[idx]
    img = read_image(img_path, mode=ImageReadMode.RGB)

    if self.transform:
      img = self.transform(img)
    return img, label

  def _build_file_list(self):
    return [(os.path.join(self.img_dir, label, file), self.labels_to_idx[label])
            for label in self.img_labels
            for file in os.listdir(os.path.join(self.img_dir, label))]


# transformations applied before training
transforms = Compose([
  Resize(size=(224, 224)),
  RandomHorizontalFlip(),
  ToDtype(dtype=torch.float32, scale=True),
  Normalize([0.6053, 0.5874, 0.5538], [0.3392, 0.3222, 0.3360])
])

# build dataset
dataset = PokemonImageDataset(img_dir=img_dir,
                              img_labels=img_labels,
                              labels_to_idx=stoi,
                              transform=transforms)

# split dataset into training and validation sets
generator = torch.Generator().manual_seed(42)
train, val = random_split(dataset, [.9, .1], generator=generator)

# intantiate dataloaders
train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)

# residual convolutional NN with 34 layers as per paper


model = ResidualConvolutionalNN(150)
m = model.to(device)
model.eval()

# function to estimate model validation error
@torch.no_grad()
def get_val_loss(val_loader):
  losses = torch.zeros(len(val_loader))
  for i, (xb, yb) in enumerate(val_loader):
    xb = xb.to(device)
    yb = yb.to(device)
    logits, loss = m(xb, yb)
    losses[i] = loss
  
  return losses.mean()

# optimizer and strategy for lr reduction
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

# train and save model
for epoch in range(num_epochs):
  for xb, yb in train_dataloader:
    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    del xb, yb, logits

  val_loss = get_val_loss(val_dataloader)
  prev_lr = optimizer.param_groups[0]['lr']
  scheduler.step(val_loss)
  cur_lr = optimizer.param_groups[0]['lr']
  if epoch % 10 == 0:
    print(f'Epoch {epoch}: train loss {loss}, validation loss {val_loss}')
    print(f'Epoch {epoch} learning rate started at {prev_lr} and stepped to {cur_lr}')

torch.save(model.state_dict(), os.path.join(os.path.abspath('./'), 'poke_classifier_params'))