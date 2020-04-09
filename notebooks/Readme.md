This folder shows experiments for SynBio with machine learning algorithms.

# Sequence Design for First Round

Three approaches:
1. Baseline sequence (20') + change one position once (61 sequence totally).
2. Random (6') (4^6 combs): [random_seq_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/random_seq_design.ipynb) 
3. Bandit recommendations based on the current experiment:  [bandit_sequence_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/bandit_sequence_design_sum_of_onehot_spectrum.ipynb) 

The bandit experiments shows our approaches better than random (not the arm space is now only limited to the available choices for sake of evaluation). 
The way recommend multiple arms is still naive: return the top n ucbs. 

We decide to design 61 (baseline) + 60 (random) + 60 (bandit). 

# Regression

- Evaluate based on the RMSE score (the smaller the better). 
- Labels are min-max normalized (between 0 and 1).
- Regression models: kernel ridge regression; Gaussian process regression.
- Kernels: 
  - Kernels from sklearn library: DotProduct; RBF
  - [String kernels](https://dx.plos.org/10.1371/journal.pcbi.1000173): spectrum, mixed spectrum, weighted degree, weighted degree with shifting

### Experiments

- regression on the first round result with cross validation: [regression](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/first_round_result_regression-Eva_on_ave.ipynb)
- [kernel analysis](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/first_round_result_regression-Eva_on_ave.ipynb)): plot kernel matrix for sequences 
- First round results: [violinplot for groups](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/Label_violinplot.ipynb)

# Recommendation for sequentail experiemntal desgin: Multi-armed Bandits

- Model: GP-UCB
- Comparison: Random selection
- Evaluation metric: expected cumulative regrets

### Experiments

[Second round rec](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/Second_round_design.ipynb): recommend second round design (60 RBS) with the optimal parameters selected from [cross validation](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/first_round_result_regression-Eva_on_ave.ipynb)
