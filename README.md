# Strategic Advantage Weighting for Imitation Learning

[Paper](https://drive.google.com/file/d/1Mr-gTc1Xu1PQNzG7JO23XxnQb6WXsQ9F/view?usp=sharing)

## Important Files


### Experiment Files

* [generate_data.py](./generate_data.py)
  * Top level script used to simulate an agent in an environment
  * Serializes trajectory data to be used for training
* [train.py](./train.py)
  * Top level script used to train a pytorch BC agent
  * See file for all available training options
* [visualization.py](./visualization.py)
  * Top level script used to analyze the results of data generation/training
  * Superimposes heat map over gridworld to visualize metric of choice
  * See file for available options and uses
* [play_gridworld.py](./play_gridworld.py)
  * Spawns interactive GUI to allow user to interact with pre-defined gridworld environments


### RL Files
* [rl.py](rl.py)
  * All utils for running tabular RL on a gridworld environment
* [deep_rl.py](./deep_rl.py)
  * code for training an on-policy, offline Q agent

