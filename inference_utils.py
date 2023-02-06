import torch
import clip
import pandas as pd
from PIL import Image


def load_csv(CSV_path):
    df = pd.read_csv(CSV_path)
    return df

def tokenize_all_caps(df, device):
    text = clip.tokenize(df["caption"].tolist(), truncate=True).to(device)
    return text

def preprocess_images(df, device, preprocess):
    images = [preprocess(Image.open(fig_path)).unsqueeze(0).to(device) for fig_path in df["fig_path"].tolist()]
    #images = torch.cat(images)
    return images

def normalize(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features
    
def grouper(n, iterable):
    iters = [iter(iterable)] * n
    return zip(*iters)

def inference(CSV_path, model, preprocess = None):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #print(f"Model: {model}")
    #print(f"Device: {device}")
    if isinstance(model, str) and preprocess is None:
        model, preprocess = clip.load(model, device=device)
        print("Running Inference using default CLIP models")
        print(f"Model: {model}")
    elif isinstance(model, torch.nn.Module) and preprocess:
        model.eval()
        #print("Running Inference using custom passed model and preprocess")
    else:
        print("Error. Incorrect model or preprocess passed")


    df = load_csv(CSV_path)
    text = tokenize_all_caps(df, device)
    images = preprocess_images(df, device, preprocess)


    with torch.no_grad():
        #print(f"Number of images: {len(images)}")
        #print(text.shape)
        image_features = torch.cat([model.encode_image(torch.cat(image_batch, 0)) for image_batch in grouper(100, images)], 0)
        image_features, text_features = normalize(image_features, model.encode_text(text))
        similarities = image_features @ text_features.T
        print(similarities)
        map = 0
        for i in range(similarities.shape[0]):
            rank = sum(similarities[i] >= similarities[i][i]).item()
            map += 1/rank
        print(f"Mean Average Precision: {map/similarities.shape[0]}")
        return map/similarities.shape[0]


        
        #logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

if __name__ == "__main__":
    CSV_PATH = "/rds/project/rds-lSmP1cwRttU/aj625/datasets/scicap_test_data/raw_caps_test.csv"
    CHECKPOINT_PATH = "/home/aj625/rds/rds-t2-cs151-lSmP1cwRttU/aj625/models/model_RN50_epoch_1_batch_1_map_0.2269.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN50", device=device)
    #checkpoint = torch.load(CHECKPOINT_PATH)
    #model.load_state_dict(checkpoint['model_state_dict'])
    inference(CSV_PATH, model=model, preprocess = preprocess)
