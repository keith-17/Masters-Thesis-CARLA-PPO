# Masters-Thesis-CARLA-PPO
This project is a collection of scripts used for my masters thesis. To develop an autonomous vehicle using deep reinforcement learning. I want to explicity make clear this project is a fork from an existing project. But due to restrictions on the school computer I couldnt effectively clone the project. 

The main reason I chose PPO is that continuous action is fundamentally how vehicle physics works. Another significant contribution is to add a recurrent neural network to actor-critic architecture. This accelerated learning.

This project is adapted from:
https://github.com/bitsauce/Carla-ppo

As seen below, the most significant experiment observation is value misalignment between the developer and the agent. Car continously driving on the road and pavement.



My own adaptations include:
1) A script to view spawn points in an environment and see what a potential route would be.
2) A script to collect image training data for perception. (no HUD)
3) Script for a Convolutional Variational Auto-Encoder (not adapted)
4) Script that interleaves training between two environments, lap environment and route environment. 
5) Environments script that contains settings for lap and route environment. Alongside essential functions for the agents custom "YUV" colour space. Then feeding back sensor data from the IMU sensor and lidar sensor. 
6) A folder that contains three different combinations Actor Critic Network that involves a recurratn neural network.
7) A changed wrappers and HUD script that allows additional sensors to fit on the vehicle object. 

Experiment 1
Comparing two representations for the observation of the agent. First using the segmentation camera with lidar data overlayed. The second to convert the RGB output to YUV format, then swap the first chroma component with the one dimensional depth sensor. (Less effort than changing the agent to observe 4 dimensions). Both representations had the autoencoder pretrained with suitable augmented data.

Experiment 2
Using the DUV colour space, compare model performance on an MLP Network to an RNN network. RNN too literally 5 attempts to stay on a road whilst the MLP took 1200 attempts.

Experiment 3
Use different combinations of Actor-Critic Network structure on the agent. Most notable emergent behavior was when only the critic had an RNN. Leading it to continuosly drive on the road and pavement. 

Experiment 4
Add Kalman filtering and an IMU sensor measurements to the observation space. Initialy comparing action spaces, adding a seperate space to break. Failed miserably. So the environments script was changed to interleave training between two environments. Overcoming catastrophic forgetting as it learnt a lot faster since had a greater deal of behaviours to methdollically learn. 
