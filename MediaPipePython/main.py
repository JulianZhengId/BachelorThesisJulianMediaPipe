import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
from pathlib import Path
from benchmarks import FilterBenchmark
from utils import LandmarksProcessor, FilesManager

# const
MAX_HANDS = 1
COLUMNS = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']]
RIGHT_HAND_MATRIX = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
])

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)


def capture_data(video_path):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    log.info('starting capture raw data...')

    cap = cv2.VideoCapture(video_path)
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=MAX_HANDS)
    detector = vision.HandLandmarker.create_from_options(options)

    frames_data = []

    filedir = Path(video_path).parent
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_path = filedir / "landmarks_video.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(image)
        new_frame_data = []

        if len(detection_result.hand_world_landmarks) > 0:
            ### visualization
            for hand_landmarks in detection_result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS, vision.drawing_styles.get_default_hand_landmarks_style(), vision.drawing_styles.get_default_hand_connections_style())

            keypoint_3d = detection_result.hand_world_landmarks[0]
            keypoint_3d_array = LandmarksProcessor.parse_keypoint_3d(keypoint_3d)
            keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
            joint_pos = keypoint_3d_array @ RIGHT_HAND_MATRIX

            ### read mediapipe joints
            for i, pos in enumerate(joint_pos):
                new_frame_data.append(np.array([pos[0], pos[1], pos[2]]))  ### new_frame_data stores all joints position in a frame
            frames_data.append(new_frame_data)

            cv2.imshow('preview', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    ### close visualizer
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    log.info('capture raw data done.')
    return np.array(frames_data)


def apply_butterworth_filter(landmarks_data:np.ndarray, cutoff=3, sampling_rate=30, order=2):
    from utils import ButterworthFilter
    log.info('starting butterworth filter...')

    num_frames = landmarks_data.shape[0]
    num_joints = landmarks_data.shape[1]

    butterworth_filter = ButterworthFilter(cutoff=cutoff, sampling_rate=sampling_rate, order=order)
    butterworth_filtered_landmarks_data = np.empty((num_frames, num_joints, 3))

    for joint_idx in range(num_joints):
        for x in range(3):
            butterworth_filtered_landmarks_data[:, joint_idx, x] = butterworth_filter(landmarks_data[:, joint_idx, x])

    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(butterworth_filtered_landmarks_data[frame_idx])
        butterworth_filtered_landmarks_data[frame_idx] @= wrist_rotation

    log.info('butterworth filter done.')
    return butterworth_filtered_landmarks_data

def apply_one_euro_filter(landmarks_data:np.ndarray):
    from utils import OneEuroFilter
    log.info('starting one euro filter...')
    num_frames = landmarks_data.shape[0]
    num_joints = landmarks_data.shape[1]
    min_cutoff = 1
    sampling_rate = 30
    beta = 0.5

    one_euro_filtered_landmarks_data = np.empty((num_frames, num_joints, 3))
    for joint_idx in range(num_joints):
        for x in range(3):
            # fresh filter for each joint/axis
            f = OneEuroFilter(sampling_rate=sampling_rate, min_cutoff=min_cutoff, beta=beta)
            for frame_idx in range(num_frames):
                one_euro_filtered_landmarks_data[frame_idx, joint_idx, x] = f(landmarks_data[frame_idx, joint_idx, x])

    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(one_euro_filtered_landmarks_data[frame_idx])
        one_euro_filtered_landmarks_data[frame_idx] @= wrist_rotation

    log.info('one euro filter done.')
    return one_euro_filtered_landmarks_data

def apply_kalman_filter(raw_data, process_noise=1e-1, measurement_noise=1e-5, fps=30):
    log.info('starting kalman filter...')
    from utils import KalmanFilterND
    num_frames = raw_data.shape[0]

    kalman_filter = KalmanFilterND(process_noise=process_noise,measurement_noise=measurement_noise,fps=fps)
    kalman_data = kalman_filter.smooth(raw_data)

    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(kalman_data[frame_idx])
        kalman_data[frame_idx] @= wrist_rotation

    log.info('kalman filter done.')
    return kalman_data


def apply_wrist_local_space_to_raw_data(landmarks_data:np.ndarray):
    num_frames = landmarks_data.shape[0]

    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(landmarks_data[frame_idx])
        landmarks_data[frame_idx] @= wrist_rotation

def vary_butterworth_cutoff_frequency(landmarks_data:np.ndarray):
    cutoff_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    acceleration_results = []
    jerk_results = []
    direction_results = []

    #compute raw metrics once
    total_raw_acceleration = 0
    total_raw_jerk = 0
    total_raw_direction = 0
    for i in range(21):
        for j in range(3):
            total_raw_acceleration += FilterBenchmark.acceleration_metric(landmarks_data[:, i, j])
            total_raw_jerk += FilterBenchmark.jerk_metric(landmarks_data[:, i, j])
            total_raw_direction += FilterBenchmark.direction_changes_metric(landmarks_data[:, i, j])

    #varying cutoff frequency
    for cutoff in cutoff_values:
        butterworth_data = apply_butterworth_filter(landmarks_data, cutoff=cutoff)

        total_mean_acceleration_butterworth = 0
        total_mean_jerk_butterworth = 0
        total_direction_changes_butterworth = 0
        for i in range(21):
            for j in range(3):
                total_mean_acceleration_butterworth += FilterBenchmark.acceleration_metric(butterworth_data[:, i, j])
                total_mean_jerk_butterworth += FilterBenchmark.jerk_metric(butterworth_data[:, i, j])
                total_direction_changes_butterworth += FilterBenchmark.direction_changes_metric(butterworth_data[:, i, j])

        acceleration_results.append(total_mean_acceleration_butterworth)
        jerk_results.append(total_mean_jerk_butterworth)
        direction_results.append(total_direction_changes_butterworth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Mean Acceleration
    axes[0].plot(cutoff_values, acceleration_results, marker='o', color='blue', label='Butterworth')
    axes[0].axhline(y=total_raw_acceleration, color='red', linestyle='--', label='Raw')
    axes[0].set_title('Mean Acceleration vs Cutoff Frequency')
    axes[0].set_xlabel('Cutoff Frequency (Hz)')
    axes[0].set_ylabel('Mean Acceleration')
    axes[0].legend()

    # Mean Jerk
    axes[1].plot(cutoff_values, jerk_results, marker='o', color='orange', label='Butterworth')
    axes[1].axhline(y=total_raw_jerk, color='red', linestyle='--', label='Raw')
    axes[1].set_title('Mean Jerk vs Cutoff Frequency')
    axes[1].set_xlabel('Cutoff Frequency (Hz)')
    axes[1].set_ylabel('Mean Jerk')
    axes[1].legend()

    # Direction Changes
    axes[2].plot(cutoff_values, direction_results, marker='o', color='green', label='Butterworth')
    axes[2].axhline(y=total_raw_direction, color='red', linestyle='--', label='Raw')
    axes[2].set_title('Direction Changes vs Cutoff Frequency')
    axes[2].set_xlabel('Cutoff Frequency (Hz)')
    axes[2].set_ylabel('Direction Changes')
    axes[2].legend()

    plt.tight_layout()
    plt.show()



def main(args):
    raw_data = None
    butterworth_data = None
    one_euro_data = None
    kalman_data = None
    filedir = None

    if args.video:
        filedir = Path(args.video).parent
        raw_data = capture_data(args.video)
        butterworth_data = apply_butterworth_filter(raw_data)
        one_euro_data = apply_one_euro_filter(raw_data)
        kalman_data = apply_kalman_filter(raw_data)

        print(raw_data.shape)

        reshaped_data_raw = raw_data.reshape(raw_data.shape[0], -1)
        FilesManager.save_landmarks_data(reshaped_data_raw, COLUMNS, filedir / "raw.csv")

    # if args.filter_only:
    #     filedir = Path(args.filter_only).parent
    #     raw_data = FilesManager.load_landmarks_data(args.filter_only)
    #     butterworth_data = apply_butterworth_filter(raw_data)
    #     one_euro_data = apply_one_euro_filter(raw_data)
    #     kalman_data = apply_kalman_filter(raw_data)

    apply_wrist_local_space_to_raw_data(raw_data)

    reshaped_data_raw_wrist = raw_data.reshape(raw_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_raw_wrist, COLUMNS, filedir / "raw_data.csv")

    reshaped_data_butterworth = butterworth_data.reshape(butterworth_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_butterworth, COLUMNS, filedir / "butterworth_data.csv")

    reshaped_data_one_euro = one_euro_data.reshape(one_euro_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_one_euro, COLUMNS, filedir / "one_euro_data.csv")

    reshaped_data_kalman = kalman_data.reshape(one_euro_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_kalman, COLUMNS, filedir / "kalman_data.csv")

    #plotting
    # filters = {
    #     'Butterworth': butterworth_data,
    #     'One Euro': one_euro_data,
    #     'Kalman': kalman_data
    # }
    #
    # for filter_name, filtered_data in filters.items():
    #     plt.plot(raw_data[:, 8, 0], label='Raw X')
    #     plt.plot(raw_data[:, 8, 1], label='Raw Y')
    #     plt.plot(raw_data[:, 8, 2], label='Raw Z')
    #     plt.plot(filtered_data[:, 8, 0], label=f'{filter_name} X')
    #     plt.plot(filtered_data[:, 8, 1], label=f'{filter_name} Y')
    #     plt.plot(filtered_data[:, 8, 2], label=f'{filter_name} Z')
    #
    #     plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    #     plt.xlabel("Frame", fontsize=10)
    #     plt.ylabel("Position", fontsize=10)
    #     plt.xticks(fontsize=10)
    #     plt.yticks(fontsize=10)
    #     plt.title(f"Index Fingertip Position Over Frames — {filter_name}", fontsize=12)
    #     plt.savefig(f'pinch_01_fingertip_{filter_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    #     plt.show()

    print("---------------------------- BENCHMARKS --------------------------------------")

    #vary_butterworth_cutoff_frequency(raw_data)
    total_mean_acceleration_raw = 0
    total_mean_acceleration_butterworth = 0
    total_mean_acceleration_one_euro = 0
    total_mean_acceleration_kalman = 0

    total_mean_jerk_raw = 0
    total_mean_jerk_butterworth = 0
    total_mean_jerk_one_euro = 0
    total_mean_jerk_kalman = 0

    total_direction_changes_raw = 0
    total_direction_changes_butterworth = 0
    total_direction_changes_one_euro = 0
    total_direction_changes_kalman = 0

    for i in range(21):
        for j in range (3):
            total_mean_acceleration_raw += FilterBenchmark.acceleration_metric(raw_data[:, i, j])
            total_mean_acceleration_butterworth += FilterBenchmark.acceleration_metric(butterworth_data[:, i, j])
            total_mean_acceleration_one_euro += FilterBenchmark.acceleration_metric(one_euro_data[:, i, j])
            total_mean_acceleration_kalman += FilterBenchmark.acceleration_metric(kalman_data[:, i, j])

            total_mean_jerk_raw += FilterBenchmark.jerk_metric(raw_data[:, i, j])
            total_mean_jerk_butterworth += FilterBenchmark.jerk_metric(butterworth_data[:, i, j])
            total_mean_jerk_one_euro += FilterBenchmark.jerk_metric(one_euro_data[:, i, j])
            total_mean_jerk_kalman += FilterBenchmark.jerk_metric(kalman_data[:, i, j])

            total_direction_changes_raw += FilterBenchmark.direction_changes_metric(raw_data[:, i, j])
            total_direction_changes_butterworth += FilterBenchmark.direction_changes_metric(butterworth_data[:, i, j])
            total_direction_changes_one_euro += FilterBenchmark.direction_changes_metric(one_euro_data[:, i, j])
            total_direction_changes_kalman += FilterBenchmark.direction_changes_metric(kalman_data[:, i, j])

    print("Total Mean Acceleration Raw Data: " + str(total_mean_acceleration_raw))
    print("Total Mean Acceleration Butterworth Data: " + str(total_mean_acceleration_butterworth))
    print("Total Mean Acceleration One Euro Data: " + str(total_mean_acceleration_one_euro))
    print("Total Mean Acceleration Kalman Data: " + str(total_mean_acceleration_kalman))

    print("------------------------------------------------------------------------------")

    print("Total Mean Jerk Raw Data: " + str(total_mean_jerk_raw))
    print("Total Mean Jerk Butterworth Data: " + str(total_mean_jerk_butterworth))
    print("Total Mean Jerk One Euro Data: " + str(total_mean_jerk_one_euro))
    print("Total Mean Jerk Kalman Data: " + str(total_mean_jerk_kalman))

    print("------------------------------------------------------------------------------")

    print("Total Direction Changes Raw Data: " + str(total_direction_changes_raw))
    print("Total Direction Changes Butterworth Data: " + str(total_direction_changes_butterworth))
    print("Total Direction Changes One Euro Data: " + str(total_direction_changes_one_euro))
    print("Total Direction Changes Kalman Data: " + str(total_direction_changes_kalman))

    print("------------------------------------------------------------------------------")

    print()

    print('Done...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MediaPipe Hand Optimization')
    parser.add_argument('--video', default=None, help='input path to a recorded video')
    #parser.add_argument('--filter-only', default=None, help='apply butterworth low-pass filter to smooth the raw data')

    main(parser.parse_args())
