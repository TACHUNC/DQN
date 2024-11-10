import os

class Save_model_logfile:
    def __init__(self,
                 algo_name:str,
                 env_name="cstr_ode", 
                 seed=0):
        
        log_name = algo_name + '_logs'
        directory_name= algo_name + '_preTrained'
        
        if not os.path.exists(log_name):
            os.makedirs(log_name)

        log_name = log_name + '/' + env_name + '/'
        if not os.path.exists(log_name):
                os.makedirs(log_name)
                
        current_num_files = next(os.walk(log_name))[2]
        run_num = len(current_num_files)
        
        # path to log file
        self.log_f_name = log_name + '/' + algo_name + '_' + env_name + "_log_" + str(run_num) + ".csv"
        
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

        directory_name = directory_name + '/' + env_name + '/'
        if not os.path.exists(directory_name):
                os.makedirs(directory_name)
        
        # path to model weighting      
        self.checkpoint_path = directory_name + algo_name + "_{}_{}_{}.pth".format(env_name, seed, run_num)
    
    
    def log_path(self):
        return self.log_f_name
    
    def model_path(self):
        return self.checkpoint_path