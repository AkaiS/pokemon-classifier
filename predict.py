import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import (Compose, Resize, ToDtype, Normalize)
import os
from model import ResidualConvolutionalNN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_labels = 150

img_dir = os.path.abspath('PokemonData')
img_labels = sorted(os.listdir(img_dir))
itos = {i: l for i, l in enumerate(img_labels)}

preprocess = Compose([
  Resize(size=(224, 224)),
  ToDtype(dtype=torch.float32, scale=True),
  Normalize([0.6053, 0.5874, 0.5538], [0.3392, 0.3222, 0.3360])
])

model = ResidualConvolutionalNN(num_labels)
model.load_state_dict(torch.load(os.path.abspath('poke_classifier_params')))
model.eval()
model = model.to(device)

def process_image(image):
  image_tensor = preprocess(read_image(image, mode=ImageReadMode.RGB))
  image_tensor = image_tensor.to(device)
  image_tensor = torch.unsqueeze(image_tensor, 0)
  return image_tensor

def predict_pokepic(processed_image):
  with torch.no_grad():
    logits, loss = model(processed_image)
    soft = torch.softmax(logits, 1)
    pred = torch.multinomial(soft, 1)
  return pred

while True:
  file_to_pred = input('What is the relative directory of the pokemon picture you want to predict?')
  pokepic_processed = process_image(file_to_pred)
  pred = predict_pokepic(pokepic_processed)

  print('This is a ', itos[int(pred[0])])