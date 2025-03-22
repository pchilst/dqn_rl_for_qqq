Algorithmic NASDAQ Trading with Reinforcement Learning
Project Overview
This project explores the application of reinforcement learning models from stable-baselines3 to predict NASDAQ ETF (QQQ) movements. After experimenting with various approaches, DQN (Deep Q-Network) models proved most effective for this trading strategy.
Key Features

Custom gym environment for financial market training
Separate models for long and short position strategies
Comprehensive validation methodology with held-out test sets
Integration of technical indicators and options data
Performance visualization through heatmaps and backtesting metrics

Methodology

Created a custom OpenAI Gym environment to define the trading action space
Tested multiple reinforcement learning algorithms (PPO, A2C, DQN)
Implemented rigorous validation to minimize overfitting
Optimized hyperparameters based on backtesting performance
Evaluated models using standard trading metrics 

Installation
bashCopypip install -r requirements.txt
Note: TA-Lib requires additional C dependencies. See TA-Lib installation guide for platform-specific instructions.
Usage

Data Preparation: Create your preliminary dataset using an API or pre-purchased data
Copypython dqn2.2_VAL_create_preliminary_data.py
Note: You'll need to add your own API keys and configure data sources
Feature Engineering: Generate training features and prepare test/validation splits
Copypython dqn2.2_create_train_and_test.py

Model Training: Train separate models for long and short positions
Copypython BT_ML_DQN2.2_VAL2024_GPU_SEP_SHORT_v1.5_alt3.py
python BT_ML_DQN2.2_VAL2024_GPU_SEP_LONG_v1.5_alt3.py


Project Structure
Copy├── data/                  # Data directory (not included in repo)
├── models/                # Saved model files
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code
│   ├── environments/      # Custom gym environments
│   ├── features/          # Feature engineering
│   └── visualization/     # Visualization utilities
├── requirements.txt       # Project dependencies
└── README.md              # This file
Future Work

Implement additional market factors
Explore ensemble approaches with traditional ML models
Optimize execution strategy and slippage modeling
