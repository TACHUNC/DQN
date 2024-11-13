### Pretrain-0, DeepDQN_enforce_1
- update_every=max_ep_len*3   
- n_updates= 32               
- n_multistart= 24
- batch_size= 48               
- buffer_size = 1500 

### Pretrain-1, DeepDQN_enforce_2
- update_every=max_ep_len*3   
- n_updates= 32               
- n_multistart= 24
- batch_size= 48               
- buffer_size = 1000 

### Pretrain-2, DeepDQN_enforce_3
- update_every=10   
- n_updates= 1               
- n_multistart= 24
- batch_size= 48               
- buffer_size = 1000 
- log_freq = 101


### Pretrain-3,DQN_1
- update_every=10   
- n_updates= 1               
- n_multistart= 24
- batch_size= 68               
- buffer_size = 10000 
- log_freq = 101

### Pretrain-4, DeepDQN_enforce_4 tanh output layer
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- mse_loss

1. There is no significant mismatch in norm before and after (no much information is being discarded).

2. Loss is however increases (with decreasing epsidoic reward)


### Pretrain-5, DeepDQN_enforce_5 linear output layer
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- mse_loss

Q: There is a clipping issue comes with the increasing loss and increasing clipping ratio.

A: Try using Huber loss suggested by deepmind? 
