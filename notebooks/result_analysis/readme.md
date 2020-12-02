- repeated_kfold_RBF_round01_nokenrelNormUnique.pickle
  - num_split = 5
  - num_repeat = 10
  - s_list = [1]
  - alpha_list= [1e-5, 1e-1, 0.5, 1, 2, 5] 
  - l_list =[6]
  - sigma_0_list = [0.5, 1, 1.5, 2, 2.5],
  - kernel_norm_flag=[True]
  - centering_flag=[False]
  - unit_norm_flag=[False]

- repeated_kfold_RBF_round01_kenrelNormUnique.pickle
  - num_split = 5
  - num_repeat = 10
  - s_list = [1]
  - alpha_list= [1e-5, 1e-1, 0.5, 1, 2, 5] 
  - l_list =[6]
  - sigma_0_list = [0.5, 1, 1.5, 2, 2.5],
  - kernel_norm_flag=[True]
  - centering_flag=[True]
  - unit_norm_flag=[True]
                                    

- repeated_kfold_RBF_round01_kenrelNormUniqueAdjust.pickle
  - num_split = 5
  - num_repeat = 10
  - s_list = [1]
  - alpha_list = [2]
  - l_list =[6]
  - sigma_0_list = [1]                                                                                                 
  - kernel_norm_flag=[True]
  - centering_flag=[True, False]
  - unit_norm_flag=[True, False]
                

- repeated_kfold_wd_shift_fast.pickle  
    Setting:   
        num_split = 5  
        num_repeat = 10  
        s_list = [0,1]  
        alpha_list= [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] + list(range(1,10))  
        l_list = [3, 6]  

- repeated_kfold_wd_shift_with_signal_std.pickle  
    test sigma0, the signal std of kernel  
    Setting:  
    num_split = 5  
    num_repeat = 10  
    s_list = [1]  
    alpha_list= [0, 1e-5, 1e-1, 1, 2, 5, 10]   
    l_list = [6]  
    sigma_0_list = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]