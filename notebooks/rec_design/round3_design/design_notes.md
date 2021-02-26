**2021-Feb-26**

##### Understand Round 3 design

Inspired by "one-hamming-distance" analysis, we want to understand the wds distance between the recommendation and the top seqs. 

**2021-Feb-23**

##### Understand Round 2 design

Task: train on round 0 + 1, test on round 2
Model: the same as the model we used to design round 2
Questions to answer: 
- does the design conforms to what we expected?
    - are the sequences with top predicted mean have high TIR scores as well? Rank matters. 
    - For the sequences with high uncertainty, how does the exploration go? Can we open up/rollout new areas?
    - what part of the results turn out to be surprising and why? Can we learn something to design a better round in the next step?


Refer to round2_prediction.ipynb, the conclusion is that the round 2 prediction did a good job in terms classification evaluation,
but the precise regression is not good.   
There are several sequences which the predictor gives a high prediction but turns out to be low. This accords to our other line of analysis about the one edit distance (one edit distance away from the top TIR sequences are low TIR ones).   
The "exploration" sequences are spread across different regions and in different clusters.   
Recommendation sequences in Rounds 1 and 2 have a similar spread?  


**2021-Feb-19**

##### Design goal of the last round design

We decide to design the next round (round 3) as the last round design. 
So we would focus on exploitation: try to find the highest possible TIR based on the known data. 
Our goal has two folds:
- Recommend sequences with higher "high TIRs", with the hope of beating the reference one. 
- Recommend sequences with higher "low TIRs", with the hope of not spending money on those low values at the last round. 

##### Design choices

We plan to select top n = 500 sequences with beta = 0, reject sequences according to some *quality control criteria*, then recommend 90 left top sequences. 

##### Quality control criteria
	
- edit distance based: reject sequences with 1 edit distance away from top TIR 
- kmer-based. 
- motif-based.  
- reject sequences in an unknown area.  
  Potential idea: define the "unknown area in terms clusters with less than 3 known sequences in terms wd distance", control the number of new rec in each unknown area/cluster to be less than 3. Yes, 3 is a random picked number.
- reject too similar sequences? 
- reject seqs with UCB < top seqs' LCB (need to define some beta)

##### Things need to check

- [ ] Mengyan: check how different parameters influence the recommendations
- [ ] Mengyan: control gpr prediction tails? or design some specific rules? 
- [ ] Maciej: check the statistical summaries of the existing data to conclude whether we should support "reject seq with 1 edit distance away from top TIR"
- [ ] Maciej: summarise good/bad patterns (for both kmer and motif) that we should/shouldn't include in the next round design