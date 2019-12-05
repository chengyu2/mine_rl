#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict

import torch
import minerl as ml
import numpy as np
from typing import Tuple, Union, List, Any

data_path = './data/experiment/MineRLObtainDiamond-v0/'
pipeline = ml.data.DataPipeline(data_path, 'MineRLObtainDiamond-v0', num_workers=1, worker_batch_size=1, min_size_to_dequeue=1)


def state_encoder(obs: OrderedDict[str, Any]) -> Tuple[np.array, np.array]:
    """

    :param obs:
    :return:
    """
    equipped_items = list(obs['equipped_items']['mainhand'].values())
    inventory = list(obs['inventory'].values())
    pov = obs['pov']

    inv_and_equip = np.array(equipped_items + inventory)

    return (inv_and_equip, pov)

def state_decoder(obs_tensor: torch.Tensor):
    # pytorch, how?
    obs_np = obs_tensor.numpy()

    return obs_np

def onehot_encoder(input: int, max_bit: int):
    onehot = np.zeros((max_bit))
    onehot[input] = 1

    return onehot

def onehot_decoder(input: List[int]):
    input = np.array(input)
    real_value = np.where(input==1)

    return real_value


def action_encoder(action: OrderedDict[str, Any]) -> Tuple[np.array, np.array]:
    """ Encode a gym action to (camera, rest_action) np.array

    :param action:
    :return:
    """
    camera = np.array(action['camera'])

    action.pop('camera')

    action_encode = list()
    # bool: 1 bit
    action_encode.append(action["attack"])
    # bool: 1 bit
    action_encode.append(action['back'])
    # one-hot: 5 bits
    action_encode = action_encode + onehot_encoder(action['craft'], 5)
    # one-hot: 8 bits
    action_encode = action_encode + onehot_encoder(action['equip'], 8)
    # bool: 1 bit
    action_encode.append(action['forward'])
    # bool: 1 bit
    action_encode.append(action['jump'])
    # bool: 1 bit
    action_encode.append(action['left'])
    # one-hot: 8 bits
    action_encode = action_encode + onehot_encoder(action['nearbyCraft'], 8)
    # one-hot: 3 bits
    action_encode = action_encode + onehot_encoder(action['nearbySmelt'], 3)
    # one-hot: 7 bits
    action_encode = action_encode + onehot_encoder(action['place'], 7)
    # bool: 1 bit
    action_encode.append(action['right'])
    # bool: 1 bit
    action_encode.append(action['sneak'])
    # bool: 1 bit
    action_encode.append(action['sprint'])

    return (camera, action_encode)


def action_decoder(camera_tensor, rest_action) -> OrderedDict[str, Union[int, List[float]]]:
    """

    :param camera_tensor:
    :param rest_action:
    :return:
    """

    action = OrderedDict()

    rest_action = rest_action.int().numpy()
    rest_action = list(rest_action)

    start = 0
    # bool: 1 bit
    action["attack"] = rest_action[start]
    start += 1

    # bool: 1 bit
    action['back'] = rest_action[start]
    start += 1

    # floats:
    action["camera"] = list(camera_tensor.numpy())
    # one-hot: 5 bits
    action['craft'] = onehot_decoder(rest_action[start:start+5])
    start += 5
    # one-hot: 8 bits
    action['equip'] = onehot_decoder(rest_action[start:start+8])
    start += 8
    # bool: 1 bit
    action['forward'] = rest_action[start]
    start += 1
    # bool: 1 bit
    action['jump'] = rest_action[start]
    start += 1
    # bool: 1 bit
    action['left'] = rest_action[start]
    start += 1
    # one-hot: 8 bits
    action['nearbyCraft'] = onehot_decoder(rest_action[start:start+8])
    start += 8
    # one-hot: 3 bits
    action['nearbySmelt'] = onehot_decoder(rest_action[start:start+3])
    start += 3
    # one-hot: 7 bits
    action['place'] = onehot_decoder(rest_action[start:start+7])
    start += 7
    # bool: 1 bit
    action['right'] = rest_action[start]
    start += 1
    # bool: 1 bit
    action['sneak'] = rest_action[start]
    start += 1
    # bool: 1 bit
    action['sprint'] = rest_action[start]
    start += 1

    return action


class DataLoader:

    def __init__(self,
                 data_path: str,
                 num_workers = 1,
                 worker_batch_size = 1,
                 min_size_to_dequeue = 1):
        self.data_path = data_path

        self.pipeline = ml.data.DataPipeline(self.data_path,
                                             'MineRLObtainDiamond-v0',
                                             num_workers=num_workers,
                                             worker_batch_size=worker_batch_size,
                                             min_size_to_dequeue=min_size_to_dequeue)

    def get_random_generator(self, num_epochs=-1):
        """

        :param num_epochs: -1 loop forever. Otherwise,  the number of epochs to iterate over.
        :return:
        """
        return self.pipeline.sarsd_iter(num_epochs=num_epochs)

