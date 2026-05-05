import matplotlib.pyplot as plt
import numpy as np
import argparse
import logging
import sys
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

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    ### close visualizer
    cap.release()
    cv2.destroyAllWindows()

    log.info('capture raw data done.')
    return np.array(frames_data)


def apply_butterworth_filter(landmarks_data:np.ndarray):
    from utils import ButterworthFilter
    log.info('starting butterworth filter...')

    num_frames = landmarks_data.shape[0]
    num_joints = landmarks_data.shape[1]

    cutoff = 3
    sampling_rate = 29.94
    order = 2

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
    min_cutoff = 2
    sampling_rate = 30
    beta = 1

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

def apply_kalman_filter_one_way(raw_data: np.ndarray, process_noise=1e-4, measurement_noise=1e-2, fps=30):
    from utils import KalmanFilterND
    num_frames, num_landmarks, num_dims = raw_data.shape

    kalman = KalmanFilterND(num_landmarks, num_dims, process_noise, measurement_noise, fps)

    filtered = np.zeros_like(raw_data)
    for frame in range(num_frames):
        filtered[frame] = kalman.update(raw_data[frame])

    return filtered

def apply_kalman_filter(raw_data, process_noise=1e-1, measurement_noise=9e-2, fps=30):
    log.info('starting kalman filter...')
    num_frames = raw_data.shape[0]
    kalman_data = raw_data.copy()
    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(kalman_data[frame_idx])
        kalman_data[frame_idx] @= wrist_rotation

    forward = apply_kalman_filter_one_way(kalman_data, process_noise, measurement_noise, fps)
    backward = apply_kalman_filter_one_way(kalman_data[::-1], process_noise, measurement_noise, fps)[::-1]
    kalman_data = (forward + backward) / 2

    log.info('kalman filter done.')
    return kalman_data


def apply_wrist_local_space_to_raw_data(landmarks_data:np.ndarray):
    num_frames = landmarks_data.shape[0]

    for frame_idx in range(num_frames):
        wrist_rotation = LandmarksProcessor.estimate_frame_from_hand_points(landmarks_data[frame_idx])
        landmarks_data[frame_idx] @= wrist_rotation

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

    if args.filter_only:
        filedir = Path(args.filter_only).parent
        raw_data = FilesManager.load_landmarks_data(args.filter_only)
        butterworth_data = apply_butterworth_filter(raw_data)
        one_euro_data = apply_one_euro_filter(raw_data)
        kalman_data = apply_kalman_filter(raw_data)

    apply_wrist_local_space_to_raw_data(raw_data)

    reshaped_data_raw_wrist = raw_data.reshape(raw_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_raw_wrist, COLUMNS, filedir / "raw_data.csv")

    reshaped_data_butterworth = butterworth_data.reshape(butterworth_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_butterworth, COLUMNS, filedir / "butterworth_data.csv")

    reshaped_data_one_euro = one_euro_data.reshape(one_euro_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_one_euro, COLUMNS, filedir / "one_euro_data.csv")

    reshaped_data_kalman = kalman_data.reshape(one_euro_data.shape[0], -1)
    FilesManager.save_landmarks_data(reshaped_data_kalman, COLUMNS, filedir / "kalman_data.csv")

    ### plotting to see difference between raw and filtered data
    plt.plot(raw_data[:, 8, 0])
    plt.plot(raw_data[:, 8, 1])
    plt.plot(raw_data[:, 8, 2])
    plt.plot(kalman_data[:, 8, 0])
    plt.plot(kalman_data[:, 8, 1])
    plt.plot(kalman_data[:, 8, 2])
    #plt.plot(one_euro_data[:, 8, 1])
    #plt.plot(kalman_data[:, 8, 1])
    plt.xlabel("Frame")
    plt.ylabel("Y position")
    plt.title("Index Fingertip Y Over Time")
    plt.show()

    butterworth_index_x = butterworth_data[:, 8, 0]
    butterworth_index_y = butterworth_data[:, 8, 1]
    butterworth_index_z = butterworth_data[:, 8, 2]

    one_euro_index_x = one_euro_data[:, 8, 0]
    one_euro_index_y = one_euro_data[:, 8, 1]
    one_euro_index_z = one_euro_data[:, 8, 2]

    print("--------------------------- ACCELERATION METRIC -------------------------------")

    raw_acceleration_x = FilterBenchmark.acceleration_metric(raw_data[:, 8, 0])
    raw_acceleration_y = FilterBenchmark.acceleration_metric(raw_data[:, 8, 1])
    raw_acceleration_z = FilterBenchmark.acceleration_metric(raw_data[:, 8, 2])

    butterworth_acceleration_x = FilterBenchmark.acceleration_metric(butterworth_index_x)
    butterworth_acceleration_y = FilterBenchmark.acceleration_metric(butterworth_index_y)
    butterworth_acceleration_z = FilterBenchmark.acceleration_metric(butterworth_index_z)

    one_euro_acceleration_x = FilterBenchmark.acceleration_metric(one_euro_index_x)
    one_euro_acceleration_y = FilterBenchmark.acceleration_metric(one_euro_index_y)
    one_euro_acceleration_z = FilterBenchmark.acceleration_metric(one_euro_index_z)

    print("Acceleration Raw Index X: " + str(raw_acceleration_x))
    print("Acceleration Butterworth Index X: " + str(butterworth_acceleration_x))
    print("Acceleration One Euro Index X: " + str(one_euro_acceleration_x))

    print("------------------------------------------------------------------------------")

    print("Acceleration Raw Index Y: " + str(raw_acceleration_y))
    print("Acceleration Butterworth Index Y: " + str(butterworth_acceleration_y))
    print("Acceleration One Euro Index Y: " + str(one_euro_acceleration_y))

    print("------------------------------------------------------------------------------")

    print("Acceleration Raw Index Z: " + str(raw_acceleration_z))
    print("Acceleration Butterworth Index Z: " + str(butterworth_acceleration_z))
    print("Acceleration One Euro Index Z: " + str(one_euro_acceleration_z))

    print()

    print("--------------------------- JERK METRIC --------------------------------------")

    raw_jerk_x = FilterBenchmark.jerk_metric(raw_data[:, 8, 0])
    raw_jerk_y = FilterBenchmark.jerk_metric(raw_data[:, 8, 1])
    raw_jerk_z = FilterBenchmark.jerk_metric(raw_data[:, 8, 2])

    butterworth_jerk_x = FilterBenchmark.jerk_metric(butterworth_index_x)
    butterworth_jerk_y = FilterBenchmark.jerk_metric(butterworth_index_y)
    butterworth_jerk_z = FilterBenchmark.jerk_metric(butterworth_index_z)

    one_euro_jerk_x = FilterBenchmark.jerk_metric(one_euro_index_x)
    one_euro_jerk_y = FilterBenchmark.jerk_metric(one_euro_index_y)
    one_euro_jerk_z = FilterBenchmark.jerk_metric(one_euro_index_z)

    print("Jerk Raw Index X: " + str(raw_jerk_x))
    print("Jerk Butterworth Index X: " + str(butterworth_jerk_x))
    print("Jerk One Euro Index X: " + str(one_euro_jerk_x))

    print("------------------------------------------------------------------------------")

    print("Jerk Raw Index Y: " + str(raw_jerk_y))
    print("Jerk Butterworth Index Y: " + str(butterworth_jerk_y))
    print("Jerk One Euro Index Y: " + str(one_euro_jerk_y))

    print("------------------------------------------------------------------------------")

    print("Jerk Raw Index Z: " + str(raw_jerk_z))
    print("Jerk Butterworth Index Z: " + str(butterworth_jerk_z))
    print("Jerk One Euro Index Z: " + str(one_euro_jerk_z))

    print()

    print("--------------------------- DIRECTIONS CHANGES -------------------------------")

    raw_sign_x = FilterBenchmark.direction_changes_metric(raw_data[:, 8, 0])
    raw_sign_y = FilterBenchmark.direction_changes_metric(raw_data[:, 8, 1])
    raw_sign_z = FilterBenchmark.direction_changes_metric(raw_data[:, 8, 2])

    butterworth_sign_x = FilterBenchmark.direction_changes_metric(butterworth_index_x)
    butterworth_sign_y = FilterBenchmark.direction_changes_metric(butterworth_index_y)
    butterworth_sign_z = FilterBenchmark.direction_changes_metric(butterworth_index_z)

    one_euro_sign_x = FilterBenchmark.direction_changes_metric(one_euro_index_x)
    one_euro_sign_y = FilterBenchmark.direction_changes_metric(one_euro_index_y)
    one_euro_sign_z = FilterBenchmark.direction_changes_metric(one_euro_index_z)

    print("Sign Raw Index X: " + str(raw_sign_x))
    print("Sign Butterworth Index X: " + str(butterworth_sign_x))
    print("Sign One Euro Index X: " + str(one_euro_sign_x))

    print("------------------------------------------------------------------------------")

    print("Sign Raw Index Y: " + str(raw_sign_y))
    print("Sign Butterworth Index Y: " + str(butterworth_sign_y))
    print("Sign One Euro Index Y: " + str(one_euro_sign_y))

    print("------------------------------------------------------------------------------")

    print("Sign Raw Index Z: " + str(raw_sign_z))
    print("Sign Butterworth Index Z: " + str(butterworth_sign_z))
    print("Sign One Euro Index Z: " + str(one_euro_sign_z))

    print()

    print("--------------------------- EFFICIENCY ---------------------------------------")


    butterworth_efficiency_x = FilterBenchmark.filter_efficiency(raw_data[:, 8, 0], butterworth_index_x)
    butterworth_efficiency_y = FilterBenchmark.filter_efficiency(raw_data[:, 8, 1], butterworth_index_y)
    butterworth_efficiency_z = FilterBenchmark.filter_efficiency(raw_data[:, 8, 2], butterworth_index_z)

    one_euro_efficiency_x = FilterBenchmark.filter_efficiency(raw_data[:, 8, 0], one_euro_index_x)
    one_euro_efficiency_y = FilterBenchmark.filter_efficiency(raw_data[:, 8, 0], one_euro_index_y)
    one_euro_efficiency_z = FilterBenchmark.filter_efficiency(raw_data[:, 8, 0], one_euro_index_z)

    print("Efficiency Butterworth Index X: " + str(butterworth_efficiency_x))
    print("Efficiency One Euro Index X: " + str(one_euro_efficiency_x))

    print("------------------------------------------------------------------------------")

    print("Efficiency Butterworth Index Y: " + str(butterworth_efficiency_y))
    print("Efficiency One Euro Index Y: " + str(one_euro_efficiency_y))

    print("------------------------------------------------------------------------------")

    print("Efficiency Butterworth Index Z: " + str(butterworth_efficiency_z))
    print("Efficiency One Euro Index Z: " + str(one_euro_efficiency_z))

    print('Done...')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MediaPipe Hand Optimization')
    parser.add_argument('--video', default=None, help='input path to a recorded video')
    parser.add_argument('--filter-only', default=None, help='apply butterworth low-pass filter to smooth the raw data')

    main(parser.parse_args())
