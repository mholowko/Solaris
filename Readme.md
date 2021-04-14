# Optimized Experimental Design for Translation Initiation using Machine Learning

## Introduction

Synthetic Biology is on the verge of a leap into high-throughput data generation for which new methods of data handling and analysis will have to be developed. In this work, we show how machine learning can be used to analyse, predict the performance of the ribosome binding site (RBS) of E. coli – one of the main genetic elements controlling protein expression. We also show how to sequentially design the RBS sequence the find the optimal choice with high protein expression as fast as possible.  

## Methods

We build a Gaussian process regression model to predict the translation initiation rate (TIR) of each gene in terms of different RBS design. We formalize sequential experiment design as a multiarmed bandit problem.  All possible unique sequences of RBS form the decision set, and the algorithm recommends design choices for each round. The experimental validation uses synthetic biology, with a plasmid inserted into E. coli. We compare our experimental design with random selections, in terms of the cumulative regret caused by not choosing the optimal sequence.

## Results

We have analysed a number of datasets available from literature guiding our choice of algorithms and encoding methods. We discuss the generation and analysis of custom data produced in the CSIRO-UQ BioFoundry.

## Summary

Machine learning is seeing increasing use in synthetic biology, where it guides more and more design decisions. In this instance we have shown how Gaussian process regression model can be used for prediction of TIR of an E. coli RBS.

This project is internally known as [SOLARIS] in CSIRO-UQ BioFoundry.

## Content

- codes: 
  * generating the data we use for machine learning algorithms (data_generating)  
  * regression (embedding, kernels_for_GPK, regression)  
  * online learning with bandits (environment, ucb)
  * others: unsorted codes

- data: 
  * baseline data 
  * Designs: design seqeucnes
  * results (First_round_results, First_round_plates)
  * generated data for regression: firstRound_4h.csv; firstRound_4h+Baseline.csv
  * RBS_natural: unlabelled data for unsupervised learning

- notebooks:
  * analysis result (result_analysis)
  * generating recommendations (rec_design)
  * nlp: nlp embedding for biological sequences
  * others: unsorted notebooks

## Run Experiments

- visualise data: run [notebook Data_visualisation](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/Data_visualisation.ipynb)  
  including label histogram, violinplot, swarmsplot

- run regression: run [notebook Compare_with_baseline](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/Compare_with_baseline.ipynb)  
  provide comparison with baseline model (GP with RBF)

- run repeated kfold: run [notebook repeated_kfold](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/repeated_kfold.ipynb) Note: depends on the setting, this can take a long time to run.  
  to show results from kfold, run [notebook plot_repeated_kfold](https://github.com/mholowko/SynbioML/blob/master/notebooks/result_analysis/plot_repeated_kfold.ipynb)

- Generate recommendations: run [notebook Batch_ucb](https://github.com/mholowko/SynbioML/blob/master/notebooks/rec_design/Batch_ucb.ipynb) with appropriate known sequences and baseline setting.
  
## Writing 

./paper  

Overleaf link https://www.overleaf.com/1948376545cbvtqtbcytyn

## Usage 

### Data pre-processing
python codes/data_generating.py normalize_flag to_design_round
normalize_flag can be True or False, default as True 
to_design_round can be integer 0-4, default as 4.

For example, if you want to generate unnormalised data for round 3's result, in your terminal type in: python codes/data_generating.py False 4

This will output file in ./data/pipeline_data

### Data visualisation
restart and run all: notebooks/result_analysis/Data_visualisation.ipynb
The ratio normalisation is in ``How about using ratio as normalisation" section
