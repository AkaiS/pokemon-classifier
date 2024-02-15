import os
from PIL import Image

path = os.path.abspath('./PokemonData')
count = 0
for folder in os.listdir(path):
  for file in os.listdir(os.path.join(path, folder)):
    img_path = os.path.join(path, folder, file)
    if not file.endswith('.jpg') and not file.endswith('.jpeg') and not file.endswith('.png'):
      os.remove(img_path)
      count += 1
    else:
      img = Image.open(img_path)
      img.info.pop('icc_profile', None)
      img.save(img_path)
print(f'Removed {count} file(s) with improper extension(s).')