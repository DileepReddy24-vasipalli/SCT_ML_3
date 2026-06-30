import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# --- ఇక్కడి నుండి రీప్లేస్ చేయండి ---
import glob

DATA_DIR = r"C:\Users\dilee\OneDrive\Desktop\SCT_ML_TASK_03"

data = []
labels = []

print("We are trying to searching for the images...")


cat_files = glob.glob(os.path.join(DATA_DIR, "**", "*cat*.*"), recursive=True)
dog_files = glob.glob(os.path.join(DATA_DIR, "**", "*dog*.*"), recursive=True)

if len(cat_files) == 0:
    cat_files = glob.glob(os.path.join(DATA_DIR, "**", "cat", "*.*"), recursive=True) + glob.glob(os.path.join(DATA_DIR, "**", "cats", "*.*"), recursive=True)
if len(dog_files) == 0:
    dog_files = glob.glob(os.path.join(DATA_DIR, "**", "dog", "*.*"), recursive=True) + glob.glob(os.path.join(DATA_DIR, "**", "dogs", "*.*"), recursive=True)

print(f"Number of cat Images: {len(cat_files)}")
print(f"Number of Dog Images: {len(dog_files)}")

for img_path in cat_files[:1000]:
    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            img_resized = cv2.resize(img_array, (64, 64))
            data.append(img_resized.flatten())
            labels.append(0)  # Cat = 0
    except Exception as e:
        pass

# 2. కుక్కల ఇమేజ్ లను ప్రాసెస్ చేయడం
for img_path in dog_files[:1000]:
    try:
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_array is not None:
            img_resized = cv2.resize(img_array, (64, 64))
            data.append(img_resized.flatten())
            labels.append(1)  # Dog = 1
    except Exception as e:
        pass

# --- ఇక్కడి వరకూ రీప్లేస్ చేయండి ---

if len(data) == 0:
    for img in os.listdir(DATA_DIR):
        try:
            if 'cat' in img:
                label = 0
            elif 'dog' in img:
                label = 1
            else:
                continue
                
            img_path = os.path.join(DATA_DIR, img)
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img_array, (64, 64))
            
            data.append(img_resized.flatten())
            labels.append(label)
        except Exception as e:
            pass


X = np.array(data) / 255.0
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

print("TRAINING is started for SVM..")

model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Model training is completed : {accuracy * 100:.2f}%")

with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("save the model as 'svm_model.pkl'")