# dmu-project

Repository for 2023 CU DMU final project.

## Models

Models were trained with PyTorch using Reinforcement Learning and a policy gradient. They are stored in the [.models](./.models) directory. Complete metrics were collected with [Aim](https://github.com/aimhubio/aim).

| Model ID                 | Short Description                                                       |
| ------------------------ | ----------------------------------------------------------------------- |
| b60ba6f06be54de99c2f890f | 200 neuron single hidden layer network with 1000 max steps per episode. |
| a53b7b3457f14f4e99172150 | 200 neuron single hidden layer network with 5000 max steps per episode. |

## Source Files

| File                | Description                                              |
| ------------------- | -------------------------------------------------------- |
| helpers.py          | Helper functions for saving and loading PyTorch models.  |
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

First make sure your environment is activated.

```sh
python -m pip install -r requirements.txt
```
