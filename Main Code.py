---------------------------------------------------
#IMPORTING MODULS
---------------------------------------------------
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

---------------------------------------------------
#GENERATING OBJECTS
---------------------------------------------------

def generate_object_data(object_type, n=100):
    data = []
    for _ in range(n):
        if object_type == 'schwarzschild':
            mass = np.random.uniform(3, 30)
            spin = 0
            fluct = np.random.uniform(0.001, 0.01)
            temp = 1 / (8 * np.pi * mass)  # Hawking temp ~ 1/M
            curvature = mass / 10
            momentum = np.random.normal(0, 0.1, 3)
            gw = 0.6
            compactness = np.random.uniform(0.3, 0.5)
            label = 0

        elif object_type == 'kerr':
            mass = np.random.uniform(5, 50)
            spin = np.random.uniform(0.1, 0.99)
            fluct = np.random.uniform(0.001, 0.02)
            temp = 1 / (8 * np.pi * mass) * (1 - spin**2)**0.5
            curvature = mass / 15
            momentum = np.random.normal(0, 0.3, 3)
            gw = 0.8
            compactness = np.random.uniform(0.35, 0.5)
            label = 1

        elif object_type == 'neutron_star':
            mass = np.random.uniform(1.2, 2.5)
            spin = np.random.uniform(0.05, 0.3)
            fluct = np.random.uniform(0.01, 0.05)
            temp = 0
            curvature = mass / 5
            momentum = np.random.normal(0, 0.5, 3)
            gw = 0.4
            compactness = np.random.uniform(0.2, 0.3)
            label = 2

        elif  object_type == 'white_dwarf':
            mass = np.random.uniform(0.5, 1.4)
            spin = np.random.uniform(0, 0.1)
            fluct = np.random.uniform(0.001, 0.02)
            temp = 0
            curvature = mass / 5
            momentum = np.random.normal(0, 0.05, 3)
            gw = 0.05
            compactness = np.random.uniform(0.01, 0.03)
            label = 3

        elif object_type == 'boson_star':  #Theoretical Object
            mass = np.random.uniform(0.01, 5)
            spin = np.random.uniform(0, 0.005)
            fluct = np.random.uniform(0.4, 0.9)
            temp = 0
            curvature = mass / 5
            momentum = np.random.normal(0, 1 , 3)
            gw = 0.35
            compactness = np.random.uniform(0.1, 0.25)
            label = 4

        elif  object_type == 'top_quark':
            mass = 0
            spin = np.random.uniform(0.4 , 0.6)
            fluct = np.random.uniform(0.6, 0.9)
            temp = 0
            curvature = 0
            momentum = np.random.normal(0, 2 , 3)
            gw = 0
            compactness = np.random.uniform(0, 0.001)
            label = 5

        elif  object_type == 'graviton_cluster':  #Theoretical Object
            mass = 0
            spin = 2
            fluct = np.random.uniform(0.01, 0.05)
            temp = 0
            curvature = 0
            momentum = np.random.normal(0, 1 , 3)
            gw = np.random.uniform(0.7 , 1)
            compactness = 0
            label = 6

        elif  object_type == 'primordial_black_hole':
            mass = 10 ** np.random.uniform(-10, 2)
            spin = np.random.uniform(0, 0.3)
            fluct = np.clip(1/(mass + 0.0000000001),0,1)
            temp = 1 / (8 * np.pi * mass)
            curvature = mass / 5
            momentum = np.random.normal(0, 0.5, 3)
            gw = np.random.uniform(0,0.8)
            compactness = np.random.uniform(0.45, 0.5)
            label = 7

        data.append([
            mass, spin, *momentum, curvature, fluct, temp, gw, compactness, label
        ])
    return np.array(data)

all_data = np.vstack([
    generate_object_data('schwarzschild', 500),
    generate_object_data('kerr', 500),
    generate_object_data('neutron_star', 500),
    generate_object_data('white_dwarf', 500),
    generate_object_data('boson_star', 500),
    generate_object_data('top_quark', 500),
    generate_object_data('graviton_cluster', 500),
    generate_object_data('primordial_black_hole', 500),
    
])

---------------------------------------------------
#TRANSFORMING THE OBJECTS INTO DATAFRAME
---------------------------------------------------

columns = ['mass', 'spin', 'p_x', 'p_y', 'p_z',
           'curvature_scalar', 'quantum_fluctuation_strength',
           'hawking_temp', 'gw_emission', 'compactness', 'label']

df = pd.DataFrame(all_data, columns=columns)
object_dataset = df.copy()

---------------------------------------------------
#MODEL TRAINING
---------------------------------------------------

X = object_dataset.drop("label", axis = 1)
y = object_dataset["label"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
rf_model = RandomForestClassifier().fit(X_train,y_train)
y_pred = rf_model.predict(X_test)

---------------------------------------------------
#MODEL EVALUATION OF NON-TUNED MODEL
---------------------------------------------------

accuracy_score(y_test,y_pred)    # Equals 1.
cross_val_score(rf_model,X_train,y_train,cv = 10).mean()     # Equals 1.

---------------------------------------------------
#MODEL TUNING
---------------------------------------------------

rf_params = {"max_depth":np.arange(1,10),
             "max_features":np.arange(1,10),
             "n_estimators":np.arange(10,1000,10),
             "min_samples_split":np.arange(1,10),}

rf_cv = GridSearchCV(rf_model,rf_params,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
rf_tuned = RandomForestClassifier(max_depth=rf_cv.best_params_["max_depth"],
                                  max_features=rf_cv.best_params_["max_features"],
                                  n_estimators=rf_cv.best_params_["n_estimators"],
                                  min_samples_split= rf_cv.best_params_["min_samples_split"]).fit(X_train,y_train)
y_pred = rf_tuned.predict(X_test)

---------------------------------------------------
#MODEL EVALUATION OF TUNED MODEL
---------------------------------------------------

accuracy_score(y_test,y_pred)  # Equals 0.999.
cross_val_score(rf_tuned,X_train,y_train,cv=10).mean()  # Equals 1.
