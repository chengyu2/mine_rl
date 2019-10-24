#!/usr/bin/env python
# -*- coding: utf-8 -*-

import minerl as ml

#ml.data.download('')

data_path = './data/experiment/MineRLObtainDiamond-v0/'
pipeline = ml.data.DataPipeline(data_path, 'MineRLObtainDiamond-v0', num_workers=1, worker_batch_size=1, min_size_to_dequeue=1)


#method 1: iterate the data in a video.
for result in pipeline.load_data('v1_absolute_grape_changeling-6_37339-46767'):
    # result is a tuple:
    #   Tuple(state, player_action, reward_from_action, next_state, is_next_state_terminal)
    print (result)

# method 2:
#pipeline.load_data('v1_absolute_grape_changeling-6_37339-46767')
for result in pipeline.sarsd_iter(num_epochs=1):
    print (result)
    break


