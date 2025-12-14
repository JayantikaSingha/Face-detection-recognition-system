from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import argparse
import pickle
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", default="C:/Users/HP/Desktop/face detection/output/embeddings.pickle",
    help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", default="C:/Users/HP/Desktop/face detection/output/recognizer.pickle",
    help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", default="C:/Users/HP/Desktop/face detection/output/le.pickle",
    help="path to output label encoder")
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

embeddings = np.array(data["embeddings"])
embeddings = embeddings.reshape(-1, len(embeddings[0]))

print("[INFO] tuning hyperparameters...")
param_grid = {'C': [1,2,3,4,5,6,7],
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

svc = SVC(probability=True)
grid_search = GridSearchCV(svc, param_grid, cv=5, n_jobs=-1)
grid_search.fit(embeddings, labels)

print("Best Parameters:", grid_search.best_params_)
best_svc = grid_search.best_estimator_

os.makedirs(os.path.dirname(args["recognizer"]), exist_ok=True)
with open(args["recognizer"], "wb") as f:
    pickle.dump(best_svc, f)

os.makedirs(os.path.dirname(args["le"]), exist_ok=True)
with open(args["le"], "wb") as f:
    pickle.dump(le, f)

print("[INFO] Model trained and saved successfully!")
