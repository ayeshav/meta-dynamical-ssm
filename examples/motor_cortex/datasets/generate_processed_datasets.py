import argparse
import os

from dandi.download import download

from processing import process_co_reaching_data, process_maze_data

DANDISETS = {
    'centre_out_reaching': '000688',
    'maze': '000070',
}


def download_data(data_root):
    for task, dandiset_id in DANDISETS.items():
        output_dir = os.path.join(data_root, task)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Downloading {task} data (dandiset {dandiset_id})...")
        download(
            urls=[f"https://dandiarchive.org/dandiset/{dandiset_id}"],
            output_dir=output_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download and preprocess motor cortex NWB datasets from DANDI."
    )
    parser.add_argument(
        '--data-root', type=str, default=os.path.dirname(os.path.abspath(__file__)),
        help='Root directory where data will be downloaded and saved (default: this script\'s directory)'
    )
    parser.add_argument(
        '--align-event', type=str, default='move_onset',
        choices=['move_onset', 'go_cue', 'target_on'],
        help='Trial alignment event (default: move_onset)'
    )
    parser.add_argument(
        '--start', type=float, default=-0.140,
        help='Trial start time relative to align event in seconds (default: -0.140)'
    )
    parser.add_argument(
        '--end', type=float, default=0.561,
        help='Trial end time relative to align event in seconds (default: 0.561)'
    )
    parser.add_argument(
        '--delay', type=float, default=0.100,
        help='Spike alignment delay in seconds (default: 0.100)'
    )
    parser.add_argument(
        '--bin-size', type=float, default=0.020,
        help='Bin size in seconds (default: 0.020)'
    )
    parser.add_argument(
        '--smooth', type=float, default=0.05,
        help='Gaussian smoothing sigma in seconds (default: 0.05)'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip DANDI download and process existing NWB files'
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_data(args.data_root)

    params = {
        'align_event': args.align_event,
        'start': args.start,
        'end': args.end,
        'delay': args.delay,
        'bin_size': args.bin_size,
        'smooth': args.smooth,
    }

    process_co_reaching_data(args.data_root, params)
    process_maze_data(args.data_root, params)


if __name__ == "__main__":
    main()
