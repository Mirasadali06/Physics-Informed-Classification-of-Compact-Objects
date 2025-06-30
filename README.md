🌀 Physics-Informed Classification of Compact Objects

A machine learning pipeline for classifying relativistic and quantum-scale compact objects using synthetic physical data.

📘 Project Description

This project aims to classify astrophysical and theoretical objects (such as black holes, neutron stars, white dwarfs, etc.) using synthetic features grounded in physical principles. We simulate objects' properties like mass, spin, momentum, spacetime curvature, quantum fluctuations, and more, and train a variety of machine learning models to accurately predict their types.

The system is entirely built upon physics-inspired synthetic data, making it both controlled and informative, while also mimicking key traits of real compact objects. This is a fusion of machine learning and theoretical physics for cosmic-scale classification.

🧬 Simulated Object Types

The following object types are included:

Schwarzschild Black Hole

Kerr Black Hole

Neutron Star

White Dwarf

Boson Star (theoretical)

Top Quark

Graviton Cluster (theoretical)

Primordial Black Hole

Each object is generated with statistically appropriate physical features such as:

mass

spin

momentum vector (px, py, pz)

curvature_scalar: proxy for spacetime curvature

quantum_fluctuation_strength

hawking_temp

gw_emission: gravitational wave profile

compactness: mass-to-radius ratio approximation

📊 Exploratory Data Analysis & Visualization

We conduct extensive EDA to validate and explore the distribution of the features:

📈 Correlation matrix heatmap (with optimized font scaling and masking)

📉 Feature importance barplots (Random Forest-based)

🌌 Histograms and KDE plots of each physical variable

🧲 Class-wise feature comparison

🧪 Confusion matrix for classification performance

🧠 Models Tried (20+ total)

A broad spectrum of machine learning models was tested:

✅ Random Forest Classifier (final selected model)

Support Vector Classifier (SVC)

Support Vector Regression (SVR)

Decision Tree Classifier

Logistic Regression

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

XGBoost

LightGBM

MLPClassifier

ExtraTrees

Gradient Boosting

AdaBoost

RidgeClassifier

SGDClassifier

and more...

🧪 Final Model: Random Forest Classifier

After hyperparameter optimization (via GridSearchCV), the final tuned model achieves:

Metric

Score

Accuracy (test set)

0.999

Cross-Val Score (CV=10)

1.000

This near-perfect performance is made possible thanks to:

Highly separable feature distributions

Carefully crafted synthetic physics data

Sufficient data per class (balanced)

🔧 Tools & Libraries

Python 3.10+

NumPy / Pandas

Matplotlib / Seaborn

scikit-learn

statsmodels

📁 File Structure (Recommendation)

project-root/
├── data/
│   └── synthetic_object_dataset.csv
├── src/
│   ├── data_generation.py
│   ├── model_training.py
│   ├── eda_visuals.py
│   └── main.py
├── results/
│   ├── plots/
│   └── models/
├── README.md
├── .gitignore
└── requirements.txt

✨ Highlights

✅ Physics-informed synthetic data generation

✅ 20+ models evaluated

✅ Perfect CV scores and robust generalization

✅ Fully explainable features rooted in real physics

✅ Clean and expandable Python implementation

🧠 Future Work

Add real astrophysical data (e.g. LIGO, Gaia) for fine-tuning

Integrate more advanced models (transformers, PINNs)

Introduce noise and adversarial perturbations for stress testing

🤝 Contributing

Pull requests and forks are welcome. Open issues if you'd like to request enhancements.

📜 License

MIT License
