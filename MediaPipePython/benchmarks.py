import numpy as np

class FilterBenchmark:
    # calculates the average of acceleration
    # ideally we want constant velocity of movements so that it looks smooth
    # lower value = smoother
    @staticmethod
    def acceleration_metric(data):
        acceleration = np.diff(data, n=2, axis=0)
        return np.mean(np.abs(acceleration))

    # smoother data have lower jerk
    @staticmethod
    def jerk_metric(data):
        jerk = np.diff(data, n=3, axis = 0)
        return np.mean(np.abs(jerk))

    # count how many times the direction changes
    # it tells you how jitter it is, higher= more jitters
    # raw jittery data could change direction almost every frame
    @staticmethod
    def direction_changes_metric(data):
        velocity = np.diff(data, axis=0)
        sign_changes = np.diff(np.sign(velocity))
        return np.count_nonzero(sign_changes)

    # it measures how efficient you filter methods are
    # filters score higher when they remove a lot of jitter with minimal changes of data
    # it tells nothing about smoothness
    @staticmethod
    def filter_efficiency(raw, filtered):
        jitter_reduction = 1 - FilterBenchmark.acceleration_metric(filtered) / FilterBenchmark.acceleration_metric(raw)
        deviation = np.sqrt(np.mean((filtered - raw)**2))
        efficiency = jitter_reduction / (deviation + 1e-8)
        return efficiency
