This folder contains notebooks which design the RBS sequences for our first round experiment, with the goal to optimize the protein expression level (expressed by translation initiation rate, TIR). The RBS sequence is 20 bps, we focus on -8 ~ -13 bps and fix others as the same as the consensus sequence (i.e. TTTAAGA+NNNNNN+TATACAT). For each base, there are 4 possibilities: A, C, G, T. So totally the feature space is 4^6 = 4096. 

For the first round experiment, we design three groups:

1) 1 by 1 changing based on the consensus sequence (61)
2) Random design, including uniformly random (equal probability of choosing each letter for each base) (30); random based on the position probability matrix (PPM) (30)
3) Bandit design. The paper [Machine Learning of Designed Translational Control Allows Predictive Pathway Optimization in Escherichia coli](https://pubs.acs.org/doi/abs/10.1021/acssynbio.8b00398) did a similar experiment and based on their data, we recommendation sequences to sample using bandit algorithms. 

To test the stability and reproducibility of the design, we design sanity checks:

1) Check whether changing the parameters of kernels by a small amount changes the recommendations dramatically.
2) Check whether permutating the data change the recommendations.
3) The kernel matrix of the Bandit Top60 vs 1by1 sequences. 
4) Clustering the Top60 recommendations (we expect there are a few clusters)
5) Check where are the 1by1 sequences in terms of clustering
6) Check the intersection of recommendations based on mu, mu+sigma, sigma (Jaccard)
