# Can crystalline polymorphism be predicted from single-molecule properties?

This repository contains the code and data accompanying the publication:

> **Itamar Wallwater et al., "Can crystalline polymorphism be predicted from single-molecule properties?"**  
> https://doi.org/10.26434/chemrxiv-2025-n71h3

The scripts provided here reproduce the analyses and figures from the paper, and can be adapted for related studies.

---

## 📄 Overview

Crystalline polymorphism, the ability of a molecule to crystallize in more than one form, poses significant challenges in materials science and pharmaceuticals.  
This work investigates whether **single-molecule descriptors and quantum chemical properties** can predict the likelihood of polymorphism without requiring prior crystallographic data.

The repository includes:

- The different dataset used in the study.
- Statistical analysis and machine learning models.
- Figure generation.

---

## 📂 Installing

```
git clone https://https://github.com/itarog/Can-crystalline-polymorphism-be-predicted-from-single-molecule-properties-
```

---

## Data

### Feature set 1

14 total features. \\
229,748 molecules. \\
2.02% polymorphic molecules. \\
File path: main/database_files/features_1_df.ftr

### Feature set 2

22 total features. \\
126,748 molecules. \\
2.48% polymorphic molecules. \\
File path: main/database_files/features_2_df.ftr

### Feature set 3

738 total features. \\
126,608 molecules. \\
2.48% polymorphic molecules. \\
File path: main/database_files/features_3_df.ftr

### Feature set 4

765 total features. \\
7,231 molecules. \\
37.07% polymorphic molecules. \\
File path: main/database_files/features_4_df.ftr

---

## Evaluation of Feature sets

### Over-sampling 

Over-sampling was implemented using the "imblearn-learn" package (http://jmlr.org/papers/v18/16-365.html)

### Under-sampling

Under-sampling was implemented manually using guiding principles found in the paper "A Review on Ensembles for the Class Imbalance Problem: Bagging-, Boosting-, and Hybrid-Based Approaches" by (10.1109/TSMCC.2011. 2161285)

### Consensus features nested cross-validation (CnCV)

CnCV was implemented using the original code in the paper "Consensus features nested cross-validation" by S.Parvandeh et al (https://doi.org/10.1093/bioinformatics/btaa046)

### Positive Unlabeled (PU) learning

PU learning was implemented using the "pulearn" package (https://github.com/pulearn/pulearn)

### Evaluation code

```
estimators_dict = {'LG': LogisticRegression(random_state=42, max_iter=1000),
                   'RF': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=100),
                   'SVM': SVC(kernel='linear', random_state=42, probability=True),
                   'Nearest neighbors' : KNeighborsClassifier(n_neighbors=15, leaf_size=30),
                   'MLP': MLPClassifier(hidden_layer_sizes=(512, 512), max_iter=500, random_state=42),
                   }

feature_set_dict = {
                    1: ['ensamble', 'os_SMOTE'],
                    2: ['ensamble', 'os_SMOTE'],
                    3: ['ensamble', 'os_SMOTE'],
                    4: ['consensus_cv'],
                    5: ['consensus_cv'],
                    }

######################################
######################################
######### Normal evaluation ##########
######################################
######################################

feature_set_results = dict()
for feature_set_num, eval_modes in feature_set_dict.items():
    results = get_feature_set_results(feature_set_num, estimators_dict, pu_learning=False, eval_methods=eval_modes, predict_method='predict_proba')
    feature_set_results[feature_set_num] = results

save_data(feature_set_results, 'full_normal_eval.pkl')

######################################
######################################
########### PU evaluation ############
######################################
######################################

feature_set_results = dict()
for feature_set_num, eval_modes in feature_set_dict.items():
    results = get_feature_set_results(feature_set_num, estimators_dict, pu_learning=True, eval_methods=eval_modes, predict_method='predict_proba')
    feature_set_results[feature_set_num] = results

save_data(feature_set_results, 'full_pu_eval.pkl') 
```

---

## Loading pre-calculated models

Pre-calculated files: \\
- main/database_files/full_normal_eval.pkl
- main/database_files/full_pu_eval.pkl
The results of the evaluation can be loaded using:

```
algo_ensamble_summary_dict = unpack_saved_results_full_by_algo('C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\full_normal_eval.pkl')
algo_ensamble_summary_dict = update_ensamble_smote(algo_ensamble_summary_dict)
algo_concv_summary_dict = unpack_saved_results_full_by_algo('C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\fset_45_normal_eval.pkl')
algo_full_summary_dict = {**algo_ensamble_summary_dict, **algo_concv_summary_dict}
us_algo_full_summary_dict = merge_eval_methods_at_feature_set_level(algo_full_summary_dict, 'ensamble', 'consensus_cv', 'combined')
os_algo_full_summary_dict = merge_eval_methods_at_feature_set_level(algo_full_summary_dict, 'os_SMOTE', 'consensus_cv', 'combined')

pu_algo_ensamble_summary_dict = unpack_saved_results_full_by_algo(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\full_pu_eval.pkl')
pu_algo_ensamble_summary_dict = update_ensamble_smote(pu_algo_ensamble_summary_dict)
pu_algo_concv_summary_dict = unpack_saved_results_full_by_algo(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\fset_45_pu_eval.pkl')
pu_algo_full_summary_dict = {**pu_algo_ensamble_summary_dict, **pu_algo_concv_summary_dict}
pu_us_algo_full_summary_dict = merge_eval_methods_at_feature_set_level(pu_algo_full_summary_dict, 'ensamble', 'consensus_cv', 'combined')
pu_os_algo_full_summary_dict = merge_eval_methods_at_feature_set_level(pu_algo_full_summary_dict, 'os_SMOTE', 'consensus_cv', 'combined')

fset_ensamble_summary_dict = unpack_saved_results_full_by_fset(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\full_normal_eval.pkl')
fset_ensamble_summary_dict = update_ensamble_smote(fset_ensamble_summary_dict)
fset_concv_summary_dict = unpack_saved_results_full_by_fset(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\fset_45_normal_eval.pkl')
fset_full_summary_dict = {**fset_ensamble_summary_dict, **fset_concv_summary_dict}

pu_fset_ensamble_summary_dict = unpack_saved_results_full_by_fset(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\full_pu_eval.pkl')
pu_fset_ensamble_summary_dict = update_ensamble_smote(pu_fset_ensamble_summary_dict)
pu_fset_concv_summary_dict = unpack_saved_results_full_by_fset(r'C:\Users\itaro\OneDrive\Documents\GitHub\Crystal_structure\fset_45_pu_eval.pkl')
pu_fset_full_summary_dict = {**pu_fset_ensamble_summary_dict, **pu_fset_concv_summary_dict}
```

---

## Reproducing figures

### Per-set tables

```
fset_method_combi = [(1, 'oversampling'), (1, 'undersampling'),
                     (2, 'oversampling'), (2, 'undersampling'),
                     (3, 'oversampling'), (3, 'undersampling'),
                     (4, 'consensus_cv'), (5, 'consensus_cv'),]
for fset_num, method in fset_method_combi:
    fset_df = get_fset_table(fset_num, fset_full_summary_dict, method)
    table_name = f'fset_{fset_num}_{method}.csv'
    fset_df.to_csv(table_name)
    pu_fset_df = get_fset_table(fset_num, pu_fset_full_summary_dict, method)
    pu_table_name = 'pu_'+table_name
    pu_fset_df.to_csv(pu_table_name)
```

### Per-algorithm tables

```
os_algo_list = ['LG', 'RF', 'Nearest neighbors'] 
for algo_name in os_algo_list:
    us_df = get_algo_table(algo_name, us_algo_full_summary_dict)
    os_df = get_algo_table(algo_name, os_algo_full_summary_dict)
    us_pu_df = get_algo_table(algo_name, pu_us_algo_full_summary_dict)
    os_pu_df = get_algo_table(algo_name, pu_os_algo_full_summary_dict)
    df_list = [os_df.loc[1:3, :], os_pu_df.loc[1:3, :], us_df.loc[1:3, :], us_pu_df.loc[1:3, :]]
    united_df = pd.concat(df_list, axis=0)
    df_list2 = [os_df.loc[4, :], os_pu_df.loc[4, :]]
    mini_df = pd.concat(df_list2, axis=1).T
    united_df = pd.concat([united_df, mini_df])
    united_df.to_csv(f'{algo_name}_algo_table.csv')

us_algo_list = ['LG', 'RF', 'Nearest neighbors', 'SVM', 'MLP'] # 
for algo_name in us_algo_list:
    us_df = get_algo_table(algo_name, us_algo_full_summary_dict)
    us_pu_df = get_algo_table(algo_name, pu_us_algo_full_summary_dict)
    df_list = [us_df.loc[1:3, :], us_pu_df.loc[1:3, :]]
    united_df = pd.concat(df_list, axis=0)
    df_list2 = [us_df.loc[4, :], us_pu_df.loc[4, :]]
    mini_df = pd.concat(df_list2, axis=1).T
    united_df = pd.concat([united_df, mini_df])
    united_df.to_csv(f'{algo_name}_algo_table.csv')
```

### Raindrop plots

```
rain_drop_plotter = RainDropPlotter()
rain_drop_plotter.gen_plot_by_fset(fset_full_summary_dict, fset_num=4, pu_summary_dict=pu_fset_full_summary_dict)
```

### Radar plots

```
eval_method = 'combined'
metrics = ['Accuracy', 'Specificty','ROC-AUC', 'Recall']
radar_plotter = RadarPlotter(metrics)
radar_plotter.gen_plot_by_eval_method(pu_os_algo_full_summary_dict, eval_method) 
```

### TBD - ESI calculations
