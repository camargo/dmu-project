# dmu-project

Repository for my Spring 2023 Decision Making Under Uncertainty final project. I investigated the performance of a reinforcement learning agent trained using a policy gradient to play Atari Pong.

## Agent Demo

Here is a demo of one of the trained agents (a53b7b) playing and winning against the computer.

https://user-images.githubusercontent.com/683355/235566498-97fada0a-9d09-41cb-87b5-6b44b51dc6ea.mp4

## Source Files

| File                | Description                                                 |
| ------------------- | ----------------------------------------------------------- |
| model_helpers.py    | Helper functions for saving and loading PyTorch models.     |
| play.py             | Fun script for playing games in the Atari gym.              |
| pong_test.py        | Test script that tests the trained Pong agent.              |
| pong_train.py       | Train script that trains Pong agent via policy gradient.    |
| visualization.ipynb | Notebook for help visualizing models, and various plotting. |

## Models

Models were trained with PyTorch using Reinforcement Learning and a policy gradient. They are stored in the [models](./models) directory. Complete metrics were collected with [Aim](https://github.com/aimhubio/aim).

| Model ID                 | Train Time  | Total Fames | Total Layers | Hidden Dim | Max Episodes | Max Steps / Episode | Gamma | Learning Rate | Batch Size | Reward-to-Go | Baseline Subtraction |
| ------------------------ | ----------- | ----------- | ------------ | ---------- | ------------ | ------------------- | ----- | ------------- | ---------- | ------------ | -------------------- |
| a53b7b3457f14f4e99172150 | 38hrs       | 335,928,545 | 3            | 200        | 20000        | 5000                | 0.99  | 0.0001        | 1          | ✅           | ❌                   |
| b60ba6f06be54de99c2f890f | 12hrs 29min | 79,861,221  | 3            | 200        | 20000        | 1000                | 0.99  | 0.0001        | 1          | ✅           | ❌                   |

## Create and Activate Environment

```sh
python3 -m venv env
source env/bin/activate
```

## Install Dependencies

```sh
python -m pip install -r requirements.txt
```

## Run Model

```sh
python pong_test.py
```
