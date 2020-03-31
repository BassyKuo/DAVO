"""
Tianwei Shen, HKUST, 2018
Common utility functions
"""
import os
import numpy as np

def complete_batch_size(input_list, batch_size):
    left = len(input_list) % batch_size
    if left != 0:
        for _ in range(batch_size-left):
            input_list.append(input_list[-1])
    return input_list


def is_valid_sample(frames, tgt_idx, seq_length):
    N = len(frames)
    tgt_drive, _ = frames[tgt_idx].split(' ')
    max_src_offset = int((seq_length - 1)/2)
    min_src_idx = tgt_idx - max_src_offset
    max_src_idx = tgt_idx + max_src_offset
    if min_src_idx < 0 or max_src_idx >= N:
        return False
    min_src_drive, _ = frames[min_src_idx].split(' ')
    max_src_drive, _ = frames[max_src_idx].split(' ')
    if tgt_drive == min_src_drive and tgt_drive == max_src_drive:
        return True
    return False

