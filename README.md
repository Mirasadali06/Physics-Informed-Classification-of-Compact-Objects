🎯 Physics-Informed Classification of Compact Objects

This project develops a machine learning system to classify astrophysical and theoretical objects based on synthetic data that reflects physical properties such as spacetime curvature, quantum fluctuations, momentum, and more.

-----------------------------------------------
📁 Dataset Description

The dataset consists of simulated astrophysical and theoretical objects, including:

Schwarzschild Black Hole

Kerr Black Hole

Neutron Star

White Dwarf

Boson Star (theoretical)

Top Quark

Graviton Cluster (theoretical)

Primordial Black Hole

-----------------------------------------------

Each object includes the following features:

mass: simulated rest mass

spin: angular momentum proxy

p_x, p_y, p_z: linear momentum components

curvature_scalar: spacetime curvature proxy

quantum_fluctuation_strength: variance in local field states

hawking_temp: Hawking radiation temperature (where applicable)

gw_emission: gravitational wave emission strength

compactness: mass-radius ratio approximation

All data is generated synthetically with controlled distributions and physics-consistent values.

-----------------------------------------------

🧠 Model Details

Algorithm: 20+ ML models tested, final model is RandomForestClassifier

Tuning: Performed with GridSearchCV

Categorical Encoding: All numerical; no encoding needed

Metrics: Accuracy, Cross-Validation Score

-----------------------------------------------

📊 Data Exploration & Visualization

To better understand the physical feature space, the following visualizations were generated:

📈 Correlation Matrix Heatmap (with font scaling to resolve overlaps)

📉 Feature Importance Barplot (via Random Forest)

🌠 KDE + Histogram overlays per feature

🧲 Class-wise Distribution Analysis

🧪 Confusion Matrix to analyze prediction breakdown by class

🔬 Models Evaluated

-----------------------------------------------

A total of 20+ classification and regression models were evaluated for performance, including:

Random Forest ✅ (best performing)

Support Vector Machines (SVC, SVR)

Decision Trees

Logistic Regression

K-Nearest Neighbors

Gaussian Naive Bayes

MLPClassifier

XGBoost / LightGBM / Gradient Boosting / AdaBoost

ExtraTrees

RidgeClassifier

SGDClassifier

All models were evaluated using cross-validation and held-out test set performance.

-----------------------------------------------

📈 Model Performance

Model Variant

Accuracy

Cross-Val Score

Base Model

1.000

1.000

Tuned Model

0.999

1.000

The exceptionally high performance is due to:

Clearly separable synthetic feature distributions

Balanced data across 8 well-defined object classes

Highly informative features based on physics principles

-----------------------------------------------

🧰 Libraries Used

Python 3.10+

NumPy / Pandas

scikit-learn

Seaborn / Matplotlib

statsmodels

-----------------------------------------------

✨ Key Highlights

Fully synthetic, physics-consistent dataset generation

Diverse ML experimentation (20+ models tested)

Feature importances and interpretability preserved

Scientific rigor in variable choice and simulation design

High-performance, generalizable Random Forest classifier

-----------------------------------------------

🔭 Future Improvements

Add noise and imperfect observations to simulate reality

Integrate real data (e.g. from LIGO, Gaia)

Explore physics-informed neural networks (PINNs, transformers)

-----------------------------------------------

🤝 Contributing

Forks and pull requests are welcome. Feel free to open issues with suggestions or improvements.

-----------------------------------------------

📜 License

MIT License
