import torch
import clip
import pandas as pd
from PIL import Image

def load_csv(CSV_path):
    df = pd.read_csv(CSV_path, nrows=50)
    return df

def tokenize_all_caps(df, device):
    text = clip.tokenize(df["caption"].tolist(), truncate=True).to(device)
    return text

def preprocess_images(df, device, preprocess):
    images = [preprocess(Image.open(fig_path)).unsqueeze(0).to(device) for fig_path in df["fig_path"].tolist()]
    images = torch.cat(images)
    return images

def normalize(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return image_features, text_features

def main(CSV_path, model="ViT-B/32"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model, device=device)

    df = load_csv(CSV_path)
    text = tokenize_all_caps(df, device)
    images = preprocess_images(df, device, preprocess)


    with torch.no_grad():
        print(images.shape)
        print(text.shape)
        image_features, text_features = normalize(model.encode_image(images), model.encode_text(text))
        similarities = image_features @ text_features.T
        print(similarities)
        map = 0
        for i in range(similarities.shape[0]):
            rank = sum(similarities[i] >= similarities[i][i]).item()
            map += 1/rank
        print(map/similarities.shape[0])


        
        #logits_per_image, logits_per_text = model(image, text)
        #probs = logits_per_image.softmax(dim=-1).cpu().numpy()

if __name__ == "__main__":
    CSV_PATH = r"C:\Users\Anchit Jain\ML_projects\CLIP_data\scicap_data\scicap_test_data\test.csv"
    main(CSV_PATH)
