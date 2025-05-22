


import numpy as np
import os
import cv2 # For video playback
import matplotlib.pyplot as plt
import glob # For finding all .npz files

def playback_npz_trajectory(npz_file_path: str, video_fps: int = 30):
    """
    Plays back a single .npz trajectory, displays video, and histograms numerical fields.

    Args:
        npz_file_path (str): Path to the .npz file.
        video_fps (int): Frames per second for video playback.
    """
    print(f"--- Processing: {os.path.basename(npz_file_path)} ---")

    try:
        data = np.load(npz_file_path)
    except Exception as e:
        print(f"Error loading {npz_file_path}: {e}")
        return

    # Extract fields
    images = data['image'] if 'image' in data else None
    states = data['state'] if 'state' in data else None
    actions = data['action'] if 'action' in data else None
    rewards = data['reward'] if 'reward' in data else None
    is_first = data['is_first'] if 'is_first' in data else None
    is_last = data['is_last'] if 'is_last' in data else None
    is_terminal = data['is_terminal'] if 'is_terminal' in data else None
    discount = data['discount'] if 'discount' in data else None

    num_steps = 0
    if actions is not None:
        num_steps = len(actions)
    elif states is not None:
        num_steps = len(states)
    elif images is not None:
        num_steps = len(images)

    if num_steps == 0:
        print("No steps found in this trajectory. Skipping.")
        return

    print(f"Trajectory Length: {num_steps} steps")

    # --- Video Playback ---
    if images is not None and images.ndim == 4 and images.shape[-1] == 3: # (T, H, W, 3) for RGB
        print("Playing back video (press 'q' to quit)...")
        # Ensure images are in uint8 for OpenCV
        if images.dtype != np.uint8:
            images = images.astype(np.uint8)

        window_name = f"Trajectory Playback - {os.path.basename(npz_file_path)}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Allows resizing

        delay_ms = int(1000 / video_fps)

        for i in range(num_steps):
            frame = images[i]
            # Add step number and flags to the frame for better context
            text = f"Step: {i}/{num_steps-1}"
            if is_first is not None and is_first[i]:
                text += " | FIRST"
            if is_last is not None and is_last[i]:
                text += " | LAST"
            if is_terminal is not None and is_terminal[i]:
                text += " | TERMINAL"
            if rewards is not None:
                text += f" | Reward: {rewards[i]:.2f}"

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print("Video playback stopped by user.")
                break
        cv2.destroyWindow(window_name)
    elif images is not None:
        print(f"Warning: 'images' field has unexpected shape {images.shape} or dtype. Skipping video playback.")
    else:
        print("No 'image' data found for video playback.")


    # --- Histogram and Information Display ---
    # plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle(f'Analysis for {os.path.basename(npz_file_path)}', fontsize=16)

    # State Histogram
    if states is not None and states.size > 0:
        ax = axes[0, 0]
        # Flatten state for histogram if it's multi-dimensional
        ax.hist(states.flatten(), bins=50, color='skyblue', edgecolor='black')
        ax.set_title('State Distribution')
        ax.set_xlabel('State Value')
        ax.set_ylabel('Frequency')
    else:
        axes[0, 0].set_title('State Distribution (No Data)')
        axes[0, 0].text(0.5, 0.5, 'No state data available', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)


    # Action Histogram
    if actions is not None and actions.size > 0:
        ax = axes[0, 1]
        # Flatten action for histogram if it's multi-dimensional
        ax.hist(actions.flatten(), bins=50, color='lightcoral', edgecolor='black')
        ax.set_title('Action Distribution')
        ax.set_xlabel('Action Value')
        ax.set_ylabel('Frequency')
    else:
        axes[0, 1].set_title('Action Distribution (No Data)')
        axes[0, 1].text(0.5, 0.5, 'No action data available', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)


    # Reward and Discount Histogram
    if rewards is not None and rewards.size > 0:
        ax = axes[1, 0]
        ax.hist(rewards, bins=20, color='lightgreen', edgecolor='black')
        ax.set_title('Reward Distribution')
        ax.set_xlabel('Reward Value')
        ax.set_ylabel('Frequency')
    else:
        axes[1, 0].set_title('Reward Distribution (No Data)')
        axes[1, 0].text(0.5, 0.5, 'No reward data available', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)

    if discount is not None and discount.size > 0:
        ax = axes[1, 1]
        # Discount is usually 1.0 or 0.0, so a bar plot might be more informative
        unique_discounts, counts = np.unique(discount, return_counts=True)
        ax.bar([str(d) for d in unique_discounts], counts, color='gold', edgecolor='black')
        ax.set_title('Discount Distribution')
        ax.set_xlabel('Discount Value')
        ax.set_ylabel('Count')
    else:
        axes[1, 1].set_title('Discount Distribution (No Data)')
        axes[1, 1].text(0.5, 0.5, 'No discount data available', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()

    # --- Episode Summary ---
    print("\n--- Episode Flags Summary ---")
    if is_first is not None:
        print(f"Number of 'is_first' steps: {np.sum(is_first)}")
    if is_last is not None:
        print(f"Number of 'is_last' steps: {np.sum(is_last)}")
    if is_terminal is not None:
        print(f"Number of 'is_terminal' steps: {np.sum(is_terminal)}")
        if np.any(is_terminal):
            terminal_idx = np.where(is_terminal)[0]
            print(f"Terminal steps at indices: {terminal_idx}")
    print("----------------------------\n")


def process_all_npz_files(directory: str, video_fps: int = 30, id: int = None):
    """
    Finds all .npz files in a directory and calls playback_npz_trajectory for each.

    Args:
        directory (str): The directory containing the .npz files.
        video_fps (int): Frames per second for video playback.
    """
    npz_files = sorted(glob.glob(os.path.join(directory, '*.npz')))

    if not npz_files:
        print(f"No .npz files found in {directory}.")
        return
    
    if id is not None:
        npz_files = [f for f in npz_files if id == int(f.split('_')[-1].split('.npz')[0])]
        print(npz_files)

    print(f"Found {len(npz_files)} .npz files in {directory}. Starting playback and analysis.")
    for npz_file in npz_files:
        playback_npz_trajectory(npz_file, video_fps)
        print("\n" + "="*50 + "\n") # Separator between trajectories


# Example Usage:
if __name__ == "__main__":
    import pathlib
    # IMPORTANT: Replace with the directory where you saved your converted .npz files
    npz_directory = pathlib.Path("~/converted_npz_demos").expanduser()

    # Ensure the directory exists
    if not os.path.isdir(npz_directory):
        print(f"Error: Directory '{npz_directory}' not found. Please run the conversion script first.")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=int, default=None)
    args = parser.parse_args()
    process_all_npz_files(npz_directory, video_fps=15, id=args.i) # You can adjust video_fps here