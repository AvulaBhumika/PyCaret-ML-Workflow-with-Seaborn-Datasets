#  PyCaret ML Workflow with Seaborn Datasets

This repository demonstrates end-to-end machine learning workflows using [PyCaret](https://pycaret.org/) with sample datasets from [Seaborn](https://seaborn.pydata.org/). The goal is to illustrate how to apply **Classification**, **Regression**, and **Clustering** models with minimal code using PyCaret's high-level APIs.


**Requirements include:**
- `pycaret`
- `seaborn`
- `pandas`
- `matplotlib` (for visualizations)

---

## 🔬 Use-Cases

### ✅ 1. Classification – Iris Dataset

Predict species using the classic iris dataset.

```python
from pycaret.classification import *
import seaborn as sns

df = sns.load_dataset('iris')
clf = setup(data=df, target='species', session_id=123, silent=True, verbose=False)
best_model = compare_models()
```

### ✅ 2. Regression – Tips Dataset

Predict tip amount based on restaurant bill and customer attributes.

```python
from pycaret.regression import *
df = sns.load_dataset('tips')
reg = setup(data=df, target='tip', session_id=123, silent=True, verbose=False)
best_model = compare_models()
```

### ✅ 3. Clustering – Penguins Dataset

Cluster similar penguins based on body characteristics.

```python
from pycaret.clustering import *
df = sns.load_dataset('penguins').dropna()
clus = setup(data=df.drop(columns='species'), session_id=123, silent=True, verbose=False)
kmeans = create_model('kmeans')
clustered = assign_model(kmeans)
```

---

## 📊 Visualizations

![image](https://github.com/user-attachments/assets/a3848c29-2904-40ca-a9ec-c3eaeda2653d)


Use PyCaret’s `plot_model()` to generate insightful plots:

```python
plot_model(best_model, plot='confusion_matrix')  # For classification
plot_model(best_model, plot='residuals')         # For regression
plot_model(kmeans, plot='cluster')               # For clustering
```

---

## 🧠 Key Features Demonstrated

- Auto preprocessing (normalization, encoding, etc.)
- Auto model comparison
- One-line model training and evaluation
- Visual performance diagnostics
- Cluster assignment

---

## 🚀 Future Work

- Export models and deploy using Flask or FastAPI
- Add hyperparameter tuning
- Extend with custom datasets


## 📄 License

This project is open-source under the [MIT License](LICENSE).
```

