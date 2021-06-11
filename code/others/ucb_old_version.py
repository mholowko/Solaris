import numpy as np
from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct

import matplotlib.pyplot as plt
from codes.environment import Rewards_env


class GPUCB():
    """Gaussian Process Regression Upper Confidence Bound Algorithm,
    applied to Synthetic Biology RBS sequence design.
    The design space (recommendation space) is the core part 4^6 =4096.
    The available sequences are from the whole space 4^20.

    Arguments
    -----------------------------------------------------------------------------
    env: instance of Rewards_env
        Attributes: rewards_dict, labels_dict, embedded, arm_features
    num_rounds: int
        total number of rounds. 
    num_arms: int
        number of arms
    num_init: int 
        number of initialization set 
    init_list: list
        list of idx of init arms

    rewards_dict: dict of list
        rewards environment
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: list of available labels
    labels_dict: dict of list
        label environment
        keys: string of embedded sequence 
            e.g. '001000100010001000100010'
        values: label (expectation of rewards list)
        sorted according to the values 
    embedded: ndarray
        num_arms x num_features
    arm_features: list
        list of arm features (string, e.g. '001000100010001000100010')
        same order as the labels dict

    bestarm_idx: int
        the arm with maximum label (assume only one arm is the best)
        TODO: consider the case for multiple best arms
    
    sample_features: list
        list of embedding features for selected arms
    sample_labels: list
        list of labels for selected arms
    sample_idx: list
        list of idx for selected arms

    ---------------For Evaluation-----------------------------------------  
    sd: int
        suboptimal draws
    r: float
        cumulative regret
    suboptimalDraws: list
        sum of draws of each sub-optimal arm 
    cumulativeRegrets: list
        Depends on the regret definition of each algorithm
        express the difference (of rewards) between drawing optimal arm 
        all the time and drawing arms based on the designed policy.  

    """
    def __init__(self, env, num_rounds, init_list=None, init_per=0.2, 
                 num_rec = 1, model = None, arm_features = None):
        """
        Parameters
        ----------------------------------------------------------------------
        env: instance of Rewards_env
            Attributes: rewards_dict, labels_dict
        num_rounds: int
            total number of rounds. 
        init_per: float
            (0,1) initiation percent of arms
        num_rec: int
            number of arms to recommend
        model: model to fit, default is None
        """

        self.env = env
        self.num_rounds = num_rounds
        self.num_arms = len(self.env.rewards_dict)
        self.init_list = init_list
        self.num_init = int(init_per * self.num_arms)
        
        self.num_rec = num_rec
        self.model = model

        self.rewards_dict = self.env.rewards_dict
        self.labels_dict = self.env.labels_dict
        self.embedded = self.env.embedded
        if arm_features == None:
            self.arm_features = self.env.arm_features
        else: 
            self.arm_features = arm_features

        self.bestarm_idx = np.argmax(list(self.labels_dict.values()))
        
        self.sample_features = []
        self.sample_labels = []
        self.sample_idxs = []
    
        self.sd = 0
        self.r = 0
        self.cumulativeRegrets = []
        self.suboptimalDraws = []
    
    def to_list(self, arms):
        """From strings to list.
        
        Parameters
        ---------------------------------------------------------------
        arm: str or list of str
            string for corresponding one hot encoding for rbs1 and rbs2
            e.g. '001000100010001000100010'
            
        Returns
        ----------------------------------------------------------------
        list or array
            e.g. [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        """
        
        if type(arms) is list:
            arms_encoding = np.zeros((len(arms), len(arms[0])))
            for i,arm in enumerate(arms):
                arms_encoding[i] = np.asarray([int(e) for e in list(arm)])
            return arms_encoding.astype(int) 
        else:
            return [int(e) for e in list(arms)]
        
    def to_string(self, code):
        """From list to string.
        
        Parameters
        -------------------------------------------
        code: list
            e.g. [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        
        Returns
        -------------------------------------------
        string
            e.g. '001000100010001000100010'
        """
        return ''.join(str(int(e)) for e in code) 

    def init_reward(self):
        """initialise arms in init_list once. 
        """
        if self.init_list == None:
            self.init_list = np.random.choice(self.num_arms, self.num_init, replace=False)
        elif type(self.init_list) == list:
            for i in self.init_list:
                self.sample(i)
        elif type(self.init_list) == dict:
            for key, values in self.init_list.items():
                idx = -1
                for i, f in enumerate(self.env.arm_features):
                    if key == f:
                        idx = i
                        break
                # if idx < 0:
                #    print('Cannot find idx for ', key)

                for reward in values:
                    self.sample_features.append(self.to_list(key))
                    self.sample_idxs.append(idx)
                    self.sample_labels.append(reward)
        else:
            print('Invalid input of init_list')

        
    def sample(self, idx):
        """sample for arm specified by idx

        Parameters
        -----------------------------
        idx: int
            the idx of arm to be sampled
        """
        self.sample_idxs.append(idx)
        features = self.arm_features[idx]
        self.sample_features.append(self.to_list(features))
        reward = self.env.sample(features)
        self.sample_labels.append(reward)

    def evaluate(self, selected_arm):
        """Evaluate the policy
            sd: sub-optimal draws
            r: regret
    
        Parameters
        -----------------------------
        selected_arm: int
            index of selected arm
        """
        if selected_arm != self.bestarm_idx:
            self.sd += 1
            bestarm_reward = self.env.sample(self.arm_features[self.bestarm_idx])
            self.r += bestarm_reward - self.sample_labels[-1]

        self.suboptimalDraws.append(self.sd)
        self.cumulativeRegrets.append(self.r)
        

class GP_BUCB(GPUCB):
    """batch version of GPUCB.
    Desautels et al. 2014 Algorithm 2
    http://jmlr.org/papers/volume15/desautels14a/desautels14a.pdf
    """

