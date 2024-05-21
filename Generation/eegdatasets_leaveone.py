import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import DataLoader
import os
from datasets import load_dataset
import pickle
import h5py
from tqdm import tqdm
import json
import open_clip

device = "cuda:2" if torch.cuda.is_available() else "cpu"
# vlmodel, preprocess = clip.load("ViT-B/32", device=device)
model_type = 'ViT-H-14'
# We use create_model_and_transforms instead of create_model_from_pretrained to use the training preprocess pipeline,
# which incorporates augmentations
# https://huggingface.co/docs/hub/en/open_clip
vlmodel, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
tokenizer = open_clip.get_tokenizer(model_type)

# Load the configuration from the JSON file
config_path = "data_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

class EEGDataset():
    def __init__(self, exclude_subject=None, subjects=None, split="train"):
        self.split = split
        self.exclude_subject = exclude_subject
        self.subjects = subjects
        self.dataset = load_dataset("Alljoined/05_125", split=self.split, cache_dir="/srv/eeg_reconstruction/shared")

        if subjects is not None:
            subject_ids = [int(sub.split('-')[1]) for sub in subjects]
            self.dataset = self.dataset.filter(lambda example: example['subject_id'] in subject_ids)

        # Load text 
        with open(config["caption_path"], 'rb') as file:
            self.text = pickle.load(file)
            
        # Calculate embeddings
        # Try to load the saved features if they exist
        
        features_filename = f'{model_type}_Alljoined1_embedding.pt'
        features_path = os.path.join(config["embedding_path"], features_filename)

        if os.path.exists(features_path):
            saved_features = torch.load(features_path)
            self.text_features = saved_features['text_features']
            self.img_features = saved_features['img_features']
        else:
            self.text_features = self.Textencoder()
            self.img_features = self.ImageEncoder()
            torch.save({
                'text_features': self.text_features.cpu(),
                'img_features': self.img_features.cpu(),
            }, features_path)
        

    def Textencoder(self):
        text = None
        with open(config["caption_path"], 'rb') as file:
            text = pickle.load(file)
        firstText = [group[0] for group in text] # We just use the first out of 5 sentences  
        text_tokens = tokenizer(firstText).to(device) # We just use the first out of 5 sentences   

        batch_size = 500
        text_features_list = []

        print("Encoding text")
        for i in tqdm(range(0, len(text_tokens), batch_size)):
            tokens_batch = text_tokens[i:i+batch_size]
            with torch.no_grad():
                text_features_batch = vlmodel.encode_text(tokens_batch)
                text_features_batch /= text_features_batch.norm(dim=-1, keepdim=True)
            
            text_features_list.append(text_features_batch)
            
        text_features = torch.cat(text_features_list, dim=0)
        return text_features
        
    def ImageEncoder(self):
        batch_size = 250  
        image_features_list = []

        with h5py.File(config["image_path"], 'r') as file:
            dataset = file["imgBrick"]
            num_images = dataset.shape[0]
            
            print("Encoding images")
            for i in tqdm(range(0, num_images, batch_size)):
                # Adjust slicing for batch processing
                batch_data = dataset[i:i + batch_size, :, :, :]
                # Preprocess images
                image_inputs = [preprocess_train(Image.fromarray(img)) for img in batch_data]
                image_inputs = torch.stack(image_inputs).to(device)

                with torch.no_grad():
                    batch_image_features = vlmodel.encode_image(image_inputs)
                    batch_image_features /= batch_image_features.norm(dim=-1, keepdim=True)
                
                image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        return image_features

    def __getitem__(self, index):
        example = self.dataset[index]
        x = torch.tensor(example['EEG']).float().detach()
        id_73k = example['73k_id']
        text = self.text[id_73k][0] # Select the first of the 5 images
        text_features = self.text_features[id_73k]
        img = id_73k # We save img as id, to be indexed from hdf5 later
        img_features = self.img_features[id_73k]
        
        return x, text, text_features, img, img_features
        
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    # data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"  # Replace with the path to your data
    train_dataset = EEGDataset(subjects=['sub-01'], split="train")
    test_dataset = EEGDataset(subjects=['sub-01'], split="test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 0
    x, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, text: {text}")            
    
        
    