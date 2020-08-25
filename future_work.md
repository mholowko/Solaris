- For the round 0, an active learning approach can be run to maximize the informativeness of the recommendations, e.g. maximizing the mutual information. Refer to [Navigating the protein fitness landscape with Gaussian processes](https://www.pnas.org/content/110/3/E193) Experimental design section.   
  Q: Whether predicted variance represents uncertainty and gives a solution to active learning?
- design the kernel with hyperparameter, e.g. [signal variance](https://drafts.distill.pub/gp/#section-4.2). We checked the hyperparameter with repeated kfold roughly (result [here](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/repeated_kfold_wd_shift_with_signal_std.pickle)).   
  Q: how to use the replicate variance to scale the kernel?
- unsupervised embedding
  - data here https://github.com/mholowko/SynbioML/tree/master/data/RBS_natural
  - similar idea as https://bair.berkeley.edu/blog/2019/11/04/proteins/
  - we can potentially propose a pre-trained model for RBS sequences and show downstream tasks (our experimental design).