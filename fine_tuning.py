# Taken from https://github.com/openai/CLIP/issues/83

import torch
import clip
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
#from inference_utils import load_csv


#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 


def main(CSV_path, SAVE_DIR, MODEL = "ViT-B/32", BATCH_SIZE = 500, EPOCHS = 2):

    device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.
    model, preprocess = clip.load(MODEL, device=device,jit=False) #Must set jit=False for training

    print(MODEL)
    print(device)

    class image_title_dataset(Dataset):
        def __init__(self, list_image_path,list_txt):

            self.image_path = list_image_path
            self.title  = clip.tokenize(list_txt, truncate = True) #you can tokenize everything at once in here(slow at the beginning), or tokenize it in the training loop.

        def __len__(self):
            return len(self.title)

        def __getitem__(self, idx):
            image = preprocess(Image.open(self.image_path[idx])) # Image from PIL module
            title = self.title[idx]
            return image,title

    dataloading_start_time = time.time()

    df = pd.read_csv(CSV_path)
    list_image_path = df["fig_path"].to_list()
    list_txt = df["caption"].to_list()
    dataset = image_title_dataset(list_image_path,list_txt)
    train_dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = False)

    print(f"Data prep time = {time.time() - dataloading_start_time} seconds")

    if device == "cpu":
        model.float()
    else :
        clip.model.convert_weights(model) # Actually this line is unnecessary since clip by default already on float16

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        print(f"Epoch : {epoch}")
        for batch in train_dataloader :
            batch_start_time = time.time()

            optimizer.zero_grad()

            images,texts = batch 

            images= images.to(device)
            texts = texts.to(device)

            logits_per_image, logits_per_text = model(images, texts)

            ground_truth = torch.arange(len(images),dtype=torch.long,device=device)

            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            print(f"total loss : {total_loss}")

            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else : 
                convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            print(f"Batch Process Time = {time.time() - batch_start_time}")

        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss,
        }, os.path.join(SAVE_DIR, "epoch_" + str(epoch) + "_model_" + "ViTB32" + ".pt"))
        
        print("model saved")
        print(f"Epoch Time = {time.time() - epoch_start_time} seconds")

if __name__ == "__main__":
    main("/rds/project/rds-lSmP1cwRttU/aj625/datasets/train.csv", "/rds/project/rds-lSmP1cwRttU/aj625/models", MODEL = "RN50", BATCH_SIZE = 500, EPOCHS = 2)


