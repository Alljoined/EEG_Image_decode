# EEG_Image_decode
Using vision-language models to decode natural image perception from non-invasive brain recordings.

## Dataset
Here we provide the code to reproduce the results of our data resource paper:</br>
"[A large and rich EEG dataset for modeling human visual object recognition][[paper_link](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub)]".</br>
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy




## Data availability
The raw and preprocessed EEG dataset, the training and test images and the DNN feature maps are available on [OSF][[osf](https://osf.io/3jk45/)]. The ILSVRC-2012 validation and test images can be found on [[ImageNet](https://www.image-net.org/download.php)]. To run the code, the data must be downloaded and placed into the following directories:

- **Raw EEG data:** `../project_directory/eeg_dataset/raw_data/`.
- **Preprocessed EEG data:** `../project_directory/eeg_dataset/preprocessed_data/`.
- **Training and test images; ILSVRC-2012 validation and test images:** `../project_directory/image_set/`.
- **DNN feature maps:** `../project_directory/dnn_feature_maps/pca_feature_maps`.



## Environment setup
1. Cloning and building from source
```
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install .
```
  2.If you would like to develop on LAVIS, it is recommended to install in editable mode:
```
pip install -e .
```

## Train
Please modify your data set path and run
```
python train_mask_img.py
```

