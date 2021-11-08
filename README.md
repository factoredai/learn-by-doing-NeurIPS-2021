# Learn by Doing - NeurIPS 2021 Competition (ROBO Track)

This repo contains the code for our submission of the [learn by doing](https://learningbydoingcompetition.github.io/) ROBO track. 

## Folder structure

- `training` contains the scripts that we used to train the models.
- `controllers` contains the `controller.py` script that, depending on the system, loads a trained model and predicts the next control action.

##  Models used

### Model 1 (Polynomial features with linear regression)
These models consist of an imitation learning set up, in which we try to learn the mapping f(X, Y) -> U from the given training data. We used polynomial feature augmentation (degree 2) on the current state variables.

Tricks used for the models involved clipping the predicted U vector and normalizing inputs.

### Model 2 (Two step linear regression) (Bumblebee only)

We trained a linear system model that predicted the next state given the current values of the state and the input. We then implemented a one-step dead-beat controller with clipping.


## How to train the models
