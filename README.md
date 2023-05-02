# dmu-project

Repository for 2023 CU DMU final project.

## Models

Models were trained with PyTorch using Reinforcement Learning and a policy gradient. They are stored in the [.models](./.models) directory. Complete metrics were collected with [Aim](https://github.com/aimhubio/aim).

| Model ID                 | Train Time  | Total Fames | Total Layers | Hidden Dim | Max Episodes | Max Steps / Episode | Gamma | Learning Rate | Batch Size | Reward-to-Go | Baseline Subtraction |
| ------------------------ | ----------- | ----------- | ------------ | ---------- | ------------ | ------------------- | ----- | ------------- | ---------- | ------------ | -------------------- |
| a53b7b3457f14f4e99172150 | 38hrs       | 335,928,545 | 3            | 200        | 20000        | 5000                | 0.99  | 0.0001        | 1          | ✅           | ❌                   |
| b60ba6f06be54de99c2f890f | 12hrs 29min | 79,861,221  | 3            | 200        | 20000        | 1000                | 0.99  | 0.0001        | 1          | ✅           | ❌                   |

## Source Files

| File                | Description                                              |
| ------------------- | -------------------------------------------------------- |
| model_helpers.py    | Helper functions for saving and loading PyTorch models.  |
| play.py             | Fun script for playing games in the Atari gym.           |
| pong_test.py        | Test script that tests the trained Pong agent.           |
| pong_train.py       | Train script that trains Pong agent via policy gradient. |
| visualization.ipynb | Notebook for helping visualizing models and plotting.    |

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
