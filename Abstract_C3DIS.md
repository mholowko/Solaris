### Introduction

Synthetic Biology is on the verge of a leap into high-throughput data generation for which new methods of data handling and analysis will have to be developed. 
In this work, we show how machine learning can be used to analyse, predict the performance of the ribosome binding site (RBS)
of E. coli – one of the main genetic elements controlling protein expression. We also show how to sequentially design the RBS
sequence the find the optimal choice with high protein expression as fast as possible.  

### Methods
We build a Gaussian process regression model to predict the translation initiation rate (TIR) of each gene in terms of
different RBS design. We formalize sequential experiment design as a multiarmed bandit problem. 
All possible unique sequences of RBS form the decision set, and the algorithm recommends design choices for each round.
The experimental validation uses synthetic biology, with a plasmid inserted into E. coli. 
We compare our experimental design with random selections, in terms of the cumulative regret caused by not choosing 
the optimal sequence.

### Results
We have analysed a number of datasets available from literature guiding our choice of algorithms and encoding methods.
We discuss the generation and analysis of custom data produced in the CSIRO-UQ BioFoundry.

### Summary
Machine learning is seeing increasing use in synthetic biology, where it guides more and more design decisions.
In this instance we have shown how Gaussian process regression model can be used for prediction of TIR of an E. coli RBS.


### Some other required items to check:

- The title of your presentation: Prediction and Sequential Experimental Design for Ribosome Binding Site of E. coli using Machine Learning
- Preferred presentation type: Oral Presentations
- A topic most suitable to your presentation: Extracting information from data
- Author's name: Mengyan Zhang (presenting author), Maciej Holowko, Cheng Soon Ong. (anyone else needs to be included?)
- Affiliations: Please list if any? 
