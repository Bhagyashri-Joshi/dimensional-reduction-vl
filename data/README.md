# Data folder

The app needs the **UCI HAR Dataset** (Human Activity Recognition Using Smartphones).

## Option A - automatic (default)
On first launch the app downloads the official archive from the UCI Machine Learning Repository
into this folder and extracts it to `data/UCI HAR Dataset/`. The extracted copy is reused afterwards.

## Option B - manual
1. Download: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Extract so that this file exists: `data/UCI HAR Dataset/train/X_train.txt`
   (the loader searches this folder recursively, so a nested folder also works).

## Option C - custom location
Set the environment variable `HAR_DATA_DIR` to any folder that contains the extracted dataset.

The dataset is **not committed** to the repository (TRD 3.3: confirm redistribution terms first).
Citation: Anguita et al., ESANN 2013 - see the References page of the app.
