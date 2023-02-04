# Taken from https://github.com/openai/CLIP/issues/111
import torch
import clip
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import inference_utils

# things to check
# rows of csv loaded, batch size, save path, number of epochs, saving name, model, shuffling (may be False for debugging purposes), 

def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

class TextCLIP(nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,text):
        return self.model.encode_text(text)
    
class ImageCLIP(nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

class image_title_dataset(Dataset):
    def __init__(self, list_image_path,list_txt):

        self.image_path = list_image_path
        self.title  = clip.tokenize(list_txt, truncate = True)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image,title

CSV_PATH = "/rds/project/rds-lSmP1cwRttU/aj625/datasets/scicap_test_data/raw_caps_test.csv" #"/rds/project/rds-lSmP1cwRttU/aj625/datasets/train.csv"
BATCH_SIZE = 128
EPOCH = 2
MODEL = "RN50"
SAVE_DIR = "/rds/project/rds-lSmP1cwRttU/aj625/models/"
BATCH_SAVE_INTERVAL = 2
CHECKPOINT_PATH = None #"/rds/project/rds-lSmP1cwRttU/aj625/models/epoch_2_model_RN50.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print(f"Number of GPUs = {torch.cuda.device_count()}")
print(torch.cuda.get_device_name())
print(MODEL)

model, preprocess = clip.load(MODEL,device=device,jit=False)
if CHECKPOINT_PATH:
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("checkpoint loaded")
    print(CHECKPOINT_PATH)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text)
model_image = torch.nn.DataParallel(model_image)

data_load_start_time = time.time()
df = pd.read_csv(CSV_PATH)
list_image_path = df["fig_path"].to_list()
list_txt = df["caption"].to_list()
dataset = image_title_dataset(list_image_path,list_txt)
train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)
print(f"Dataloading Time = {time.time() - data_load_start_time}")
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=5e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.05) #We chain the optimizer to model, not to model_image or model_text, since the computaional graph is still connected. So updating model will also update the model_image and model_text
if CHECKPOINT_PATH:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

for epoch in range(EPOCH):
    epoch_start_time = time.time()
    print(f"Epoch {epoch}")
    batch_number = 0

    for batch in train_dataloader :
        
        optimizer.zero_grad()
        images,texts = batch 

        #images= images.to(device)
        #texts = texts.to(device)

        image_embedding = model_image(images)
        text_embedding = model_text(texts)

        logit_scale = model.logit_scale.exp()
        logits_per_image, logits_per_text = create_logits(image_embedding,text_embedding,logit_scale)
        ground_truth = torch.arange(images.shape[0]).to(device) # this one still need manually to put on GPU
        img_loss = loss_img(logits_per_image,ground_truth)
        text_loss = loss_txt(logits_per_text,ground_truth)
        total_loss = (img_loss + text_loss )/2
        print(f"total loss : {total_loss}")
        print(f"img loss : {img_loss}")
        print(f"text loss : {text_loss}")
        total_loss.backward()
        optimizer.step()
        clip.model.convert_weights(model)

        if batch_number%BATCH_SAVE_INTERVAL == 0:
            map = inference_utils.inference("/rds/project/rds-lSmP1cwRttU/aj625/datasets/scicap_test_data/raw_caps_test.csv", model, preprocess)
            torch.save({
            'batch': batch_number,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'map': map
            }, os.path.join(SAVE_DIR,"model_" + MODEL.replace('\\', '') + "_epoch_" + str(epoch) + "_batch_" + str(batch_number) + "_map_" + str(map)[:6] + ".pt"))
            print("model saved")
        batch_number += 1

    map = inference_utils.inference("/rds/project/rds-lSmP1cwRttU/aj625/datasets/scicap_test_data/raw_caps_test.csv", model, preprocess)
    torch.save({
    'batch': batch_number,
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'map': map
    }, os.path.join(SAVE_DIR,"model_" + MODEL.replace('\\', '') + "_epoch_" + str(epoch) + "_batch_" + str(batch_number) + "_map_" + str(map)[:6] + ".pt"))
    print("model saved")
    print(f"Epoch Time = {time.time() - epoch_start_time}")
