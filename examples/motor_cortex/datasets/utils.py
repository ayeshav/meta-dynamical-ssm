import numpy as np
from scipy.signal import decimate
from scipy.ndimage import gaussian_filter1d


def get_onset_id(velocity, start_time, stop_time, go_cue_time):
    """
    Estimate movement onset
    Code adapted from https://github.com/KordingLab/DAD/blob/master/code/utils/ff_trial_table_co.m
    """
    onset_times = []

    for i in range(len(start_time)):

        start = start_time[i]
        stop = min(stop_time[i], start + 6)

        t = np.where((velocity[:, 0] > start) & (velocity[:, 0] <= stop))[0]

        speed = gaussian_filter1d(np.sqrt(velocity[t, 1] ** 2 + velocity[t, 2] ** 2), 10, mode='constant')
        ac = np.diff(speed) * 25
        d_ac = np.diff(gaussian_filter1d(ac, 10, mode='constant'))

        peaks = (d_ac[:-1] > 0) & (d_ac[1:] < 0)

        try:
            mvt_peak = np.where(peaks & (velocity[t, 0][3:] > go_cue_time[i]) & (ac[2:] > 1))[0][0]
        except:
            mvt_peak = np.where((velocity[t, 0][3:] > go_cue_time[i]) & (ac[2:] > 1))[0][0]

        thresh = ac[mvt_peak] / 2

        onset_id = np.where((ac < thresh) & (t[1:] < t[mvt_peak]))[0][-1]

        onset_times.append(velocity[t[onset_id], 0])

    return onset_times


def get_bins(event_time, start, end, binsize, end_event=None):
    if end_event is None:
        bins = [np.arange(event_time[i] + start, event_time[i] + end, binsize) for i in range(len(event_time))]
    else:
        bins = [np.arange(event_time[i] + start, end_event[i] + end, binsize) for i in range(len(event_time))]
    return bins

def get_bins_rtt(event_time, start, end, binsize):
    bins = [np.arange(event_time[i][0] + start, event_time[i][0] + end, binsize) for i in range(len(event_time))]
    return np.stack(bins)

def get_spike_counts(spike_times, bins):
    return np.histogram(spike_times, bins)[0]


def get_masked_variable(var, bins):
    mask = (var[:, 0] >= bins[0]) & (var[:, 0] < bins[1])
    return var[mask, 1:]


def make_obs_mask(bins, obs_interval):
    mask = np.full(bins.shape, False)

    for i, (start, end) in enumerate(obs_interval):
        mask[i, (bins[i][0] < start) & (bins[i][1] > end)] = True

    return mask


def get_trialized_data(neurons, behavior, all_bins, all_bins_delay, trial_length, task='maze', binsize=0.02,
                       smooth_factor=2.5):
    spike_count_trial = []
    velocity_trial = []

    for i in range(len(all_bins)):
        spike_counts = [get_spike_counts(neurons[j], all_bins_delay[i]) for j in neurons.index.values]

        beh = get_masked_variable(behavior, (all_bins[i][0].round(2), all_bins[i][-1].round(2)))

        if task == 'maze':
            beh = np.gradient(beh, axis=0) * 1_000

        ds_beh = decimate(beh, np.round(binsize/np.diff(behavior[:2, 0]).item()).astype(int), axis=0)[:trial_length]

        spike_count_trial.append(np.dstack(spike_counts))
        velocity_trial.append(ds_beh)

    spike_count_trial = np.vstack(spike_count_trial)
    velocity_trial = np.stack(velocity_trial)
    rates_trial = gaussian_filter1d(spike_count_trial.astype("float64"), smooth_factor, axis=1, mode='constant')

    return spike_count_trial, rates_trial, velocity_trial


def get_trialized_data_rtt(neurons, behavior, go_cue_time_array, all_bins, all_bins_delay, trial_length, binsize=0.02,
                           smooth_factor=2.5):
    spike_count_trial = []
    velocity_trial = []
    go_cue_trial = np.zeros((len(all_bins), trial_length, 1))

    for i in range(len(all_bins)):
        spike_counts = [get_spike_counts(neurons[j], all_bins_delay[i]) for j in neurons.index.values]
        beh = get_masked_variable(behavior, (all_bins[i][0].round(2), all_bins[i][-1].round(2)))

        ds_beh = decimate(beh, np.round(binsize/np.diff(behavior[:2, 0]).item()).astype(int), axis=0)[:trial_length]

        spike_count_trial.append(np.dstack(spike_counts))
        velocity_trial.append(ds_beh)

        go_cue_id = [np.where(all_bins[i] <= go_cue_time_array[i][j])[0][-1] for j in range(len(go_cue_time_array[i]))]
        go_cue_trial[i, np.array(go_cue_id)] = 1

    spike_count_trial = np.vstack(spike_count_trial)
    velocity_trial = np.stack(velocity_trial)
    rates_trial = gaussian_filter1d(spike_count_trial.astype("float64"), smooth_factor, axis=1, mode='constant')

    return spike_count_trial, rates_trial, velocity_trial, go_cue_trial
