from train import train
from plots import plot_training_curve, plot_reward_variance

# Run multiple seeds
results_1 = train(seed=0, reward_step=-1.0)
results_2 = train(seed=1, reward_step=-1.0)
results_3 = train(seed=2, reward_step=-1.0)

plot_reward_variance([results_1, results_2, results_3])
