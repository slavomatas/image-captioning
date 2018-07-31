import sys

sys.path.append('/opt/cocoapi/PythonAPI')
sys.path.append('/usr/local/lib/python3.6/dist-packages')
from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
from torch.autograd import Variable
from image_captioning.show_attend_tell import ShowAttendTellModel

# TODO #1: Define a transform to pre-process the testing images.
transform_test = transforms.Compose([
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

transform_test = transforms.Compose([
    transforms.CenterCrop(224),
    # transforms.RandomHorizontalFlip(), # do we need to flip when eval?
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


#-#-#-# Do NOT modify the code below this line. #-#-#-#

# Create the data loader.
data_loader = get_loader(transform=transform_test,
                         mode='test')


import numpy as np
import matplotlib.pyplot as plt

# Obtain sample image before and after pre-processing.
#next(iter(data_loader))

iterator = iter(data_loader)
orig_image, image = next(iterator)
#orig_image, image = next(iterator)
#orig_image, image = next(iterator)

# Visualize sample image, before pre-processing.
plt.imshow(np.squeeze(orig_image))
plt.title('example image')
plt.show()


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
import torch

# TODO #2: Specify the saved models to load.
model_file = "show-attend-tell-3.pkl"

# TODO #3: Select appropriate values for the Python variables below.
embed_size = 512
hidden_size = 512

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

# Initialize the encoder and decoder, and set each to inference mode.
model = ShowAttendTellModel(hidden_size, embed_size, vocab_size, embed_size)
model.eval()

# Load the trained weights.
model.load_state_dict(torch.load(os.path.join('./models', model_file)))

# Move models to GPU if CUDA is available.
model.to(device)

# Move image Pytorch Tensor to GPU if CUDA is available.
image = image.to(device)

state = (Variable(torch.zeros(image.size(0), hidden_size), volatile=True),
         Variable(torch.zeros(image.size(0), hidden_size), volatile=True))

# Pass the embedded image features through the model to get a predicted caption.
output_ids = model.sample(image, state)
text = [data_loader.dataset.vocab.idx2word[index]+" " for index in output_ids]
print(text)

