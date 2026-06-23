import os

import numpy as np
import torch
from pynwb import NWBHDF5IO

from utils import get_bins, get_trialized_data

_CO_EVENT_COL = {
    'move_onset': 'move_onset_time',
    'go_cue':     'go_cue_time',
    'target_on':  'target_on_time',
}

_MAZE_EVENT_COL = {
    'move_onset': 'move_begins_time',
    'go_cue':     'go_cue_time',
    'target_on':  'target_presentation_time',
}


def _find_nwb_files(root):
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith('.nwb'):
                yield os.path.join(dirpath, f), f


def process_co_reaching_data(data_root, params):
    datapath = os.path.join(data_root, 'centre_out_reaching')
    savepath = os.path.join(data_root, 'centre_out_reaching', 'processed')
    os.makedirs(savepath, exist_ok=True)

    align_event = params['align_event']
    start = params['start']
    end = params['end']
    delay = params['delay']
    binsize = params['bin_size']
    smooth_factor = params['smooth'] / binsize
    trial_length = int((end - start) // binsize)

    event_col = _CO_EVENT_COL[align_event]

    data_co = []
    for filepath, filename in _find_nwb_files(datapath):
        if '_ses-CO-' not in filename:
            continue
        print(f'Processing {filename}')
        with NWBHDF5IO(filepath, 'r') as io:
            nwbfile = io.read()

            trials = nwbfile.trials.to_dataframe()
            correct_trial_index = (
                (trials.result == 'R') &
                ~np.isnan(trials.target_id) &
                ((trials.stop_time - trials.start_time) < 6.0)
            )
            correct_trials = trials[correct_trial_index]

            neurons = nwbfile.units.to_dataframe().spike_times
            ts = nwbfile.processing['behavior'].containers['Velocity'].time_series['cursor_vel'].timestamps[()]
            velocity = nwbfile.processing['behavior'].containers['Velocity'].time_series['cursor_vel'].data[()]
            velocity = np.hstack((ts[:, np.newaxis], velocity))

            all_bins = get_bins(correct_trials[event_col].values, start, end, binsize)
            all_bins_delay = get_bins(correct_trials[event_col].values, start - delay, end - delay, binsize)

            spikes, rates, velocity = get_trialized_data(
                neurons, velocity, all_bins, all_bins_delay, trial_length,
                task='centre_out', binsize=binsize, smooth_factor=smooth_factor,
            )

            angle = correct_trials['target_dir'].values
            target_id = correct_trials['target_id'].values

        data_co.append({
            'y': torch.from_numpy(spikes).float(),
            'rates': torch.from_numpy(rates).float(),
            'velocity': torch.from_numpy(velocity).float(),
            'target': torch.from_numpy(target_id.astype(int)),
            'angle': torch.from_numpy(angle).float(),
            'sub': filename[:-4],
        })

    torch.save(data_co, os.path.join(savepath, f'centre_out_{align_event}_bin_size_{binsize}.pt'))


def process_maze_data(data_root, params):
    datapath = os.path.join(data_root, 'maze')
    savepath = os.path.join(data_root, 'maze', 'processed')
    os.makedirs(savepath, exist_ok=True)

    align_event = params['align_event']
    start = params['start']
    end = params['end']
    delay = params['delay']
    binsize = params['bin_size']
    smooth_factor = params['smooth'] / binsize
    trial_length = int((end - start) // binsize)

    event_col = _MAZE_EVENT_COL[align_event]

    data_maze = []
    for filepath, filename in _find_nwb_files(datapath):
        print(f'Processing {filename}')
        with NWBHDF5IO(filepath, 'r') as io:
            nwbfile = io.read()

            trials = nwbfile.trials.to_dataframe()
            try:
                correct_trial_index = (trials['correct_reach'] == 1) & ~(trials['discard_trial'] == 1)
            except KeyError:
                correct_trial_index = (trials['task_success'] == 1) & ~(trials['discard_trial'] == 1)
            correct_trials = trials[correct_trial_index]

            neurons = nwbfile.units.to_dataframe().spike_times
            ts = nwbfile.processing['behavior'].containers['Position'].spatial_series['Hand'].timestamps[()]
            pos = nwbfile.processing['behavior'].containers['Position'].spatial_series['Hand'].data[()]
            pos = np.hstack((ts[:, np.newaxis], pos))

            bins = get_bins(
                correct_trials[event_col].values, start, end + binsize, binsize,
                correct_trials['stop_time'].values,
            )
            bins_delay = get_bins(
                correct_trials[event_col].values, start - delay, end - delay, binsize,
                correct_trials['stop_time'].values,
            )

            y, rates, velocity = get_trialized_data(
                neurons, pos, bins, bins_delay, trial_length, smooth_factor=smooth_factor,
            )

            target_pos = correct_trials['hit_target_position'].values
            angle = np.stack([np.arctan2(*target[::-1]) for target in target_pos])

            try:
                trial_version = correct_trials['trial_version'].values
            except KeyError:
                trial_version = correct_trials['trial_type'].values

        data_maze.append({
            'y': torch.from_numpy(y).float(),
            'rates': torch.from_numpy(rates).float(),
            'velocity': torch.from_numpy(velocity).float(),
            'angle': angle,
            'trial_version': torch.from_numpy(trial_version).float(),
            'subject': filename[:-4],
            'task': 'maze',
        })

    torch.save(data_maze, os.path.join(savepath, f'maze_{align_event}_bin_size_{binsize}.pt'))
