# RL-for-path-planning
In this repository we will create simple reinforcement learning algorithm for path planning in different robotic settings
Continuous Action Robotic Arm: 

This repository demonstrates continuous action reinforcement learning in the specific context of robotic arms moving through a configuration space.  

 

<h3>Abstract Motivation:</h3> 

The vast majority of robotic arms used in real-world applications today are hard-coded to follow a certain trajectory when doing a specific task. In dynamic situations in which obstacles may come in the way of a specific trajectory or a specific target of the arm moves positions, such arms will fail to perform adequately. 

 

In response this repository demonstrates the potential for deep reinforcement learning as a potential avenue for creating robotic arms that can operate in dynamic environments.  

 

<h3>Continuous Action Deep Reinforcement Learning as applied to Robotic arms:</h3> 

To understand the concept of deep reinforcement learning one must first understand: 
<ol>
	<li>Robotics configuration spaces</li>
   	<li>Deep neural networks</li>  
	<li>Discrete reinforcement learning</li>  
	<li>Continuous action reinforcement learning</li>
</ol>  


<h3>The Basics of Robotic Configuration Spaces:</h3> 

A robotic configuration space plots the different positions of actuators that are possible in an environment with obstacles. For instance, consider the following robotic arm and the following configuration space.  

Robotic arm: 
<img src="images/robotic_arm_diram.png" width=200 height=200> 
![Robotic arm that is originally at point A. The goal state is point B. It's first joint angle is alpha, and its second joint angle is beta|40%](images/robotic_arm_diagram.png|width=200)

  

Configuration space: 
![C space of the above robotic arm. The shaded parts of the diagram are values for which the robot will crash into an obstacle. The unshaded parts are areas for which the robot moves freely.](images/c_space_diagram.png)
 

The shaded area above represents points at which the arm, with that particular alpha and beta values, would crash into the obstacle 

The unshaded area represents all the points at which the arm would represents points at which the robot, with that particular alpha and particular alpha and beta values, would not be crashing into the obstacle. 

 

Deep Neural Networks: 

Deep neural networks. I’m going to keep this very high level. A deep neural network is a function approximator that uses multiple levels of neurons to simulate a mathematical function. A neural network will be used as the mathematical tool for us to implement our algorithm. 
