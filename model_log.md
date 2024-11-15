### DQN_1
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64              
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001

### Deep_DQN_1
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64              
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001  
- lr = 1e-3

### Deep_DQN_2
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64              
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_3
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= `84`              
- buffer_size = 10000
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_4
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= `84`              
- buffer_size = `30000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_5
- update_every = `5`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `84`              
- buffer_size = 10000
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_6
- update_every = `5`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `84`              
- buffer_size = `30000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`
  
### Deep_DQN_7
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= `128`              
- buffer_size = `5000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = `0.005` 
- lr = `1e-4`

### Deep_DQN_8
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= `128`              
- buffer_size = `5000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_9
- update_every = 10   
- n_updates= 1               
- n_multistart= 12
- batch_size= `128`              
- buffer_size = 10000
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`


### Deep_DQN_10
- update_every = `20`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `256`              
- buffer_size = 10000
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip
- tau = 0.001 
- lr = `1e-4`

### Deep_DQN_11
- update_every = `15`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `180`              
- buffer_size = `15000`
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip
- tau = `0.002` 
- lr = `1e-4`
  
### Deep_DQN_12
- update_every = `20`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `256`              
- buffer_size = `15000`
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip
- tau = `0.002` 
- lr = `1e-4`
- `use_prioritized_buffer`


## Discover
1. smaller `lr` stablise the loss. Large TD-error values generate correspondingly large parameter update.
2. bigger `batch_size` reduces the vairance(oscillation) in reward.

3. decreasing loss, either model learn the data realy good (no suprised by the data)/potentially overfittig ro