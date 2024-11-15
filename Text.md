### Pretrain-0, DeepDQN_enforce_0
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- tanh output layer
- mse_loss

1. There is no significant mismatch in norm before and after (no much information is being discarded).

2. Loss is however increases (with decreasing epsidoic reward)


### Pretrain-1, DeepDQN_enforce_1 
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss

Q: There is a clipping issue comes with the increasing loss and increasing clipping ratio.

A: Try using Huber loss suggested by deepmind?  and update the way of calculating norm before and afters. We found that one activation layer is missing.


### Pretrain-2, DeepDQN_enforce_2
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- huber_loss
  
Q: Despite the increasing rate of loss and norm is less than the previous tasks, there are still lots of information being clipped.
A: Using autoclip

### Pretrain-3, DeepDQN_enforce_3
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip

### Pretrain-4, DQN_1
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- huber_loss
- auto_clip

### Pretrain-5, DQN_2
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip

### Pretrain-6, Deep_DQN_1
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001 
  
### Pretrain-7, Deep_DQN_2
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64               
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = `0.005` 

### Pretrain-8, Deep_DQN_3
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= `128`              
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001

### Pretrain-9, Deep_DQN_4
- update_every = 5   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64              
- buffer_size = `30000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001
  
### Pretrain-10, Deep_DQN_5
- update_every = `10`   
- n_updates= 1               
- n_multistart= 12
- batch_size= 64              
- buffer_size = 10000  
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001

### Pretrain-11, Deep_DQN_6
- update_every = `10`   
- n_updates= 1               
- n_multistart= 12
- batch_size= `84`              
- buffer_size = `30000`
- log_freq = 101
- linear output layer
- mse_loss
- auto_clip
- tau = 0.001
  

# Update
1. Examin tau and in `Pretrain7`. Test `Relection1` in `Pretrain7`.
   - loss becomes greater. (Not sure if it is the special case.)
   - The loss doesnt converge even epsilon has been converged for a long time.
2. Eaxmin batch_size in `Pretrain8`. Maybe a case of overfitting, because agent is trained in a frequent manner, it stucks at local optimum and generalise worse when more data come in. Try either increase the update interval (decrease the frequency) or increse the buffer size more
   
# TODO
1. Increases the batch size. [suggested by both DeepMind and some people](https://www.reddit.com/r/MachineLearning/comments/1fqqfos/d_batch_size_vs_learning_rate/#) 
   
2. ~~increase the tau in soft update (maybe its the mismatch in models make the loss biger and biger).~~`Pretrain7`
   
3. increase the buffer_size, similar to `TODO-1`, By the law of big number.

# Question
1. If `TODO-2` is the solution, then clipping the gradient norm should be a solution, but it is not, as the above experiment signal the result. -> Its the gradient explosion/vanishment. From the target norm before, its more likely to te explosion.
   - No, increasing tau is more likely to be the solution.  Because clip the grad norm only limits step of updation, because of the limitation, it makes the difference grow if small tau is used.
    -

2. The amplitude of loss and grad_norm_befores' oscillations remain high in most experiemtns. `Pretrain6`

3. The loss grows exponentially with time. `Pretrain6`

4. So as the large deviation in episodic rewards in some tasks at the end of the task. `Pretrain6`

# Reflection
1. If its the mismatch in model make the loss significantly different (answer to exponetial loss growth), then the model should eventually be stable at the end of the task. Because epsilon and independency in buffer is small enough to oscillate the performance. Amswer to `Q1` might be wrong. `Pretrain6`
   
2. But the mismatch in model should only be the cause of large `Loss`, but **not the reason** of large deviation in `average rewward`.