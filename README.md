# Classification on Dry Bean and Raisin (AIN3101)

Machine Learning project - Badr University in Assiut, School of AI & Data Management, Fall 2026.

We compare **Logistic Regression, Random Forest and SVM** on two UCI datasets
(3 models x 2 datasets = 6 experiments), with the same split, the same cross-validation
and the same metrics for every model.

## Team
- Mohamed Abdalrasoul
- Renad Ahmed
- Maryam Tarek

## Datasets
| Dataset | Source | License |
|---------|--------|---------|
| Dry Bean (13,611 rows, 7 classes) | Koklu, M. and Özkan, I.A. (2020). Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques. *Computers and Electronics in Agriculture*, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507 - UCI id=602, https://doi.org/10.24432/C50S4B | CC BY 4.0 |
| Raisin (900 rows, 2 classes) | Cinar, I., Koklu, M. and Tasdemir, S. (2020). Classification of Raisin Grains Using Machine Vision and Artificial Intelligence Methods. *Gazi Journal of Engineering Sciences*, 6(3), 200-209. https://doi.org/10.30855/gmbd.2020.03.03 - UCI id=850, https://archive.ics.uci.edu/dataset/850 | CC BY 4.0 |

The xlsx files are in `data/`. If they are missing, download the zip files from the UCI pages above
and put `Dry_Bean_Dataset.xlsx` in `data/dry_bean/` and `Raisin_Dataset.xlsx` in `data/raisin/`.

## How to run
```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook
```
Run the notebooks **in this order** (Kernel -> Restart & Run All):
1. `01_dry_bean.ipynb` - Dry Bean, stages 1 to 9 (about 6 minutes, most of it is the Random Forest grid search)
2. `02_raisin.ipynb` - Raisin, stages 1 to 9 (about 1 minute)
3. `03_comparison.ipynb` - comparison figures, stability check, answers to the comparison questions (under 1 minute)

Notebooks 01 and 02 save their tables to `results/tables/`, and notebook 03 reads them.
All figures are saved to `results/figures/` with their figure number.

## Project structure
```
ML-Project/
├── README.md
├── requirements.txt
├── data/
│   ├── dry_bean/Dry_Bean_Dataset.xlsx
│   └── raisin/Raisin_Dataset.xlsx
├── notebooks/
│   ├── 01_dry_bean.ipynb
│   ├── 02_raisin.ipynb
│   └── 03_comparison.ipynb
├── src/
│   └── utils.py          shared pipelines, grids, metrics and plots
├── results/
│   ├── figures/          fig01 ... fig21
│   └── tables/           results, overfitting and stability tables (csv) + best parameters (json)
└── report/
    └── report.pdf
```

The report uses its own figure numbers (Figure 1-8). Each caption names the file it comes from,
for example Figure 5 = `results/figures/fig07_drybean_cm_svm.png`.

## Main results (test set, 20%)
| Dataset | Best model | Accuracy | F1 (macro) |
|---------|-----------|----------|------------|
| Dry Bean | SVM (rbf, C=10, gamma=scale) | 0.924 | 0.935 |
| Raisin | Logistic Regression (C=0.1) | 0.878 | 0.877 |

Tested with Python 3.13, scikit-learn 1.9, pandas 3.0, numpy 2.5, matplotlib 3.11.
