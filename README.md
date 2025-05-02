# WeaSEL++: Weak Supervision via Gradient-Guided Coreset Selection

This repository provides a complete implementation of **WeaSEL++**, an advanced framework for training classification models using weak supervision and gradient-based coreset selection. It builds upon the original [WeaSEL (NeurIPS 2021)](https://arxiv.org/abs/2110.07681) and integrates RETRIEVE, a coreset algorithm that improves training sample efficiency and model performance.

---

## 🔧 Main Script

### `weasel_pipeline.py`

This is the **main file** to run the entire pipeline.

It includes:
- Data loading and preprocessing via PyTorch Lightning `DataModule`
- Training and evaluation of:
  - **WeaSEL baseline**
  - **Snorkel baseline**
  - **WeaSEL++** (our enhanced model)
- Performance comparison with visual plots

---

## 📦 Requirements

Before running the pipeline, make sure to install the required packages:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `pytorch-lightning`
- `hydra-core`
- `snorkel`
- `scikit-learn`
- `matplotlib`
- `torch`

---

## 🚀 Running the Pipeline

Make sure your working directory is set correctly and run:

```bash
python weasel_pipeline.py
```

---

## 📁 Project Structure

```
.
├── weasel_pipeline.py         # 🚀 Main script to run full pipeline
├── configs/                   # 🔧 Hydra configuration files
├── weasel/                    # 📦 Source code for WeaSEL, WeaSEL++, data loaders
├── README.md                  # 📘 This file
```

---

## 📊 Results

The script produces visual comparisons of:
- **WeaSEL++ vs Snorkel**
- **WeaSEL++ vs WeaSEL**

Evaluated using metrics such as:
- Accuracy
- F1 Score
- Precision
- Recall
- AUC

---

## 📄 Citation

If you use this codebase, please consider citing the original paper:

```
@inproceedings{weasel2021,
  title={WeaSEL: Weak Supervision for End-to-end Learning},
  author={Maheshwari, Gaurav and Ratner, Alexander and Brunk, Ben and Shankar, Vaishaal and others},
  booktitle={NeurIPS},
  year={2021}
}
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.
