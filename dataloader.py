#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict

import minerl as ml
import numpy as np
from typing import Tuple, Union, List

data_path = './data/experiment/MineRLObtainDiamond-v0/'
pipeline = ml.data.DataPipeline(data_path, 'MineRLObtainDiamond-v0', num_workers=1, worker_batch_size=1, min_size_to_dequeue=1)


def state_encoder(obs) -> Tuple[np.array, np.array]:
    """

    :param obs:
    :return:
    """
    equipped_items = list(obs['equipped_items']['mainhand'].values())
    inventory = list(obs['inventory'].values())
    pov = obs['pov']

    inv_and_equip = np.array(equipped_items + inventory)

    return (inv_and_equip, pov)

def state_decoder(obs_tensor):
    pass
    # pytorch, how?
    #obs_np = obs_tensor.to_numpy()


def action_encoder(action) -> Tuple[np.array, np.array]:
    """ Encode a gym action to (camera, rest_action) np.array

    :param action:
    :return:
    """
    camera = np.array(action['camera'])

    action.pop('camera')
    rest_action = np.array(action.values())

    return (camera, rest_action)


def action_encoder(camera_tensor, rest_action) -> OrderedDict[str, Union[int, List[float]]]:
    """

    :param camera_tensor:
    :param rest_action:
    :return:
    """

    action = OrderedDict()

    rest_action = rest_action.to_numpy()
    action["attack"] = rest_action[0]
    action['back'] = rest_action[1]

    action["camera"] = list(camera_tensor.to_numpy())

    action['craft'] = rest_action[2]
    action['equip'] = rest_action[3]
    action['forward'] = rest_action[4]
    action['jump'] = rest_action[5]
    action['left'] = rest_action[6]
    action['nearbyCraft'] = rest_action[7]
    action['nearbySmelt'] = rest_action[8]
    action['place'] = rest_action[9]
    action['right'] = rest_action[10]
    action['sneak'] = rest_action[11]
    action['sprint'] = rest_action[12]

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

