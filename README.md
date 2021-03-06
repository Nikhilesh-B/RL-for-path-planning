# RL-for-path-planning
In this repository we will create simple reinforcement learning algorithm for path planning in different robotic settings
Continuous Action Robotic Arm: 

This repository demonstrates continuous action reinforcement learning in the specific context of robotic arms moving through a configuration space.  

 

Abstract Motivation: 

The vast majority of robotic arms used in real-world applications today are hard-coded to follow a certain trajectory when doing a specific task. In dynamic situations in which obstacles may come in the way of a specific trajectory or a specific target of the arm moves positions, such arms will fail to perform adequately. 

 

In response this repository demonstrates the potential for deep reinforcement learning as a potential avenue for creating robotic arms that can operate in dynamic environments.  

 

Continuous Action Deep Reinforcement Learning as applied to Robotic arms: 

To understand the concept of deep reinforcement learning one must first understand: 

Robotics configuration spaces  

Deep neural networks  

Discrete reinforcement learning  

Continuous action reinforcement learning  

 

The Basics of Robotic Configuration Spaces: 

A robotic configuration space plots the different positions of actuators that are possible in an environment with obstacles. For instance, consider the following robotic arm and the following configuration space.  

Robotic arm: 

 

 

 

Configuration space: 

 

The shaded area above represents points at which the arm, with that particular alpha and beta values, would crash into the obstacle 

The unshaded area represents all the points at which the arm would represents points at which the robot, with that particular alpha and particular alpha and beta values, would not be crashing into the obstacle. 

 

Deep Neural Networks: 

Deep neural networks. I’m going to keep this very high level. A deep neural network is a function approximator that uses multiple levels of neurons to simulate a mathematical function. A neural network will be used as the mathematical tool for us to implement our algorithm. 
