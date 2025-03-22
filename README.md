For this project, I was curious how well reinforcement learning could predict the NASDAQ (ticker 'QQQ'). 

I began with PPO reinforcement learning models, and after doing some trial and error, I landed on DQN as being far preferable. I used gym to create a custom action space for the reinforcement models to specify exactly how they should learn. 

I held out both test and validation data to try to eliminate as much bias as possible, and created heat maps and other visualizations to try to hone in on the best models. I then saved those out for further use.

To use this project:
1. Create your preliminary dataset using an API or pre-purchased data. It doesn't necessarily need to be 'QQQ'. It should have standard open - high - low - close - volume. In my case, I also merged in option related data.
2. Run the dqn2.2_create_train_and_test.py script. This creates all the variables that we will need for training the models.
3. Run each of these below scripts, which create models for both long and short positions separately:
   BT_ML_DQN2.2_VAL2024_GPU_SEP_SHORT_v1.5_alt3.py
   BT_ML_DQN2.2_VAL2024_GPU_SEP_LONG_v1.5_alt3.py
