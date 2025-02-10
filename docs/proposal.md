---
layout: default
title: Proposal
---

## Summary of the Project
Our goal is to create a AI that can earn high score in a game called Aimlabs, which is a fps training game. 

Currently, we consider the input for our AI is the screen capturing during the game, and the output should be a model that is able to take actions to earn scores based on the in game screen capturings.

The application of this project includes making reliable AI teammates or more challenging opponents in fps games, and real-time self-tracking device that is able to efficiently locate specific objects on the screen. For instance, drones that can locate fire automatically captured by its camera.

## AI/ML Algorithms
We are going to use reinforcement learning with Proximal Policy Optimization or Advantage Actor-Crtic together with screen capture and keyboard/mouse control to complete this project.

## Evaluation Plan
Aimlabs is a fps training game. During the game, multiple objects will appears on different position, player need to move their mouse onto the object and click on them to shoot the object and earn scores. As a successful AI, it should be able to quickly move the mouse to the objects and click on them. We hope the successful AI could reach 100k score.

Since scores are all what AI care about, the higher score AI can earn, the better it performs. We expect our AI can only move randomly at the beginning of project. However, as it improves from reinforcement learning, it should learns to move mouse toward the objects as quickly as possible then click on them, that will allows it to earn the highest score. This is out moonshot case.

To test whether there is a overfit situation, we may train our module using several background, but testing it using different environment. At first, the test cases will be easy, but the difficuly will gradually increase, so that we can see the overall performance of our AI.

## Meet the Instructor
Date: 2025/1/20 10:15am

## AI Tool Usage
We are going to use ***chatgpt*** for debugging our code or optimize the speed of our code if possible.

We plan to use ***Gymnasium***, an API standard for reinforcement learning, as a basic API to connect our Aimlabs game. Meanwhile, Aimlabs has its own API we can use to extract the data for analysis purpose. Some online API tools are available for this game, too.
