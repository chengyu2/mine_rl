#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import minerl as ml
import numpy as np

#ml.data.download('')

data_path = './data/experiment/MineRLObtainDiamond-v0/'
pipeline = ml.data.DataPipeline(data_path, 'MineRLObtainDiamond-v0', num_workers=1, worker_batch_size=1, min_size_to_dequeue=1)


#method 1: iterate the data in a video.
for result in pipeline.load_data('v1_absolute_grape_changeling-6_37339-46767'):
    # result is a tuple:
    #   Tuple(state, player_action, reward_from_action, next_state, is_next_state_terminal)
    status = result[0]
    # equipped_items = [ status['equipped_items']['mainhand']['damage'], # int32
    #                    status['equipped_items']['mainhand']['maxDamage'],  # int32
    #                    status['equipped_items']['mainhand']['type'] # int32
    #                  ]

    # inventory = [ status['inventory']['coal'], # int64
    #               status['inventory']['cobblestone'], # int64
    #               status['inventory']['crafting_table'], # int64
    #               status['inventory']['dirt'], # int64
    #               status['inventory']['furnace'], # int64
    #               status['inventory']['iron_axe']
    #             ]

    equipped_items = list(status['equipped_items']['mainhand'].values())
    inventory = list(status['inventory'].values())
    pov = status['pov']

    break

    print (result)

# method 2:
#pipeline.load_data('v1_absolute_grape_changeling-6_37339-46767')
for result in pipeline.sarsd_iter(num_epochs=1):
    print (result)

