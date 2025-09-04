# Attention versus Contrastive Learning of Tabular Data: A Data-Centric Benchmarking

This repository contains the code and experiments for our paper:  

**Rabbani, S.B., Medri, I.V. & Samad, M.D. _Attention versus contrastive learning of tabular data: a data-centric benchmarking_. Int J Data Sci Anal (2024).**  
📄 [https://doi.org/10.1007/s41060-024-00681-z](https://doi.org/10.1007/s41060-024-00681-z)

---

## 📂 Repository Structure

### Baselines
- `tabnet/*` – TabNet baseline experiments  
- `NPT/*` – Neural Processes for Tabular (NPT) experiments  

### Notebooks
- `baseline-*.ipynb` – Baseline methods used in the benchmark (TabNet, NPT, FTT, etc.).  
  - ⚠️ TabNet, NPT, and FTT require their own conda environments.  
- `test-scarf-*.ipynb` – Contrastive learning experiments with different corruption (augmentation) methods.  

### Helpers
- `helpers/trainer*.py` – Training and evaluation functions for contrastive learning.  
- `helpers/corruptor_df.py` – Implementations of different corruption (augmentation) methods.  

---

## 🚀 Getting Started

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
