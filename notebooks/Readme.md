This folder shows experiments for SynBio with machine learning algorithms.

# Sequence Design for First Round

Three approaches:
1. Baseline sequence (20') + change one position once (61 sequence totally).
2. Random (6') (4^6 combs): [random_seq_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/random_seq_design.ipynb) 
3. Bandit recommendations based on the current experiment:  [bandit_sequence_design](https://github.com/mholowko/SynbioML/blob/master/notebooks/bandit_sequence_design.ipynb) 

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

[Regression for RBS - Predict FC](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_FC.ipynb):
Kernel ridge regression with one-hot embedding with DotProduct is the best (in terms of test RMSE ~0.15).

[Regression for RBS - Predict TIR](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_TIR.ipynb):
Kernel ridge regression with label embedding with WD(shift) Kenel is the best (in terms of test RMSE ~ 0.23).

With cross-validation:
[Regression for RBS - Predict F - CV](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Regression_RBS_FC%20_CV.ipynb)

Cross Prediction using two dataset:
[generate_rbs_rec_with_cross_prediction](https://github.com/mholowko/SynbioML/blob/master/notebooks/generate_rbs_rec_with_cross_prediction.ipynb)

# Recommendation for sequentail experiemntal desgin: Multi-armed Bandits

- Model: GP-UCB
- Comparison: Random selection
- Evaluation metric: expected cumulative regrets

### Experiments

[Recommend RBS sequences FC](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Recommend%20RBS%20sequences%20FC.ipynb)  
[Recommend RBS sequences TIR](https://github.com/chengsoonong/eheye/blob/master/SynBio/notebooks/Recommend%20RBS%20sequences%20TIR.ipynb)

For both cases, GP-UCB has smaller cumulative regrets compared with random selection.

Interesting facts:
  - For TIR, regression is worse, but recommendations with ucb are better. (potential bugs?)
  - regret is in linear rate, rather than log rate (which is not ideal?)
  
TODO:
  - Recommend more than one arm once (subset selection)
  - PWM calculation
  - Unsupervised regresentation 
 
