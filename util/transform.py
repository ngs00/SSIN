import numpy
import cv2


class AddNoise(object):
    def __init__(self, p=0.2, target_snr_db=[2, 10], mean_noise=0):
        self.p = p
        self.snr = target_snr_db
        self.mean_noise = mean_noise

    def __call__(self, signal):
        if self.p > numpy.random.rand():
            noise = numpy.random.uniform(self.snr[0], self.snr[1], (1, signal.shape[1])).astype(numpy.float32)
            signal = signal + noise
            signal[signal < 0] = 0
            signal[signal > 1] = 1

        return signal


class MaskZeros(object):
    def __init__(self, p=0.2, mask_p=[0.1, 0.3]):
        self.p = p
        self.mask_p = mask_p

    def __call__(self, signal):
        if self.p > numpy.random.rand():
            _, signal_len = signal.shape
            target_mask_p = numpy.random.uniform(self.mask_p[0], self.mask_p[1])
            mask_size = int(target_mask_p * signal_len)
            target_mask = numpy.random.randint(0, signal_len-1, mask_size)
            signal[:, target_mask] = 0.0
        return signal


class ShiftLR(object):
    def __init__(self, p=0.2, shift_p=[0.01, 0.05]):
        self.p = p
        self.shift_p = shift_p

    def __call__(self, signal):
        if self.p > numpy.random.rand():
            _, signal_len = signal.shape
            target_shift_p = numpy.random.uniform(self.shift_p[0], self.shift_p[1])
            shift_size = int(target_shift_p * signal_len)
            shift_signal = numpy.zeros_like(signal)
            if numpy.random.rand() > 0.5:
                shift_signal[:, shift_size:] = signal[:, :-shift_size]
            else:
                shift_signal[:, :-shift_size] = signal[:, shift_size:]
            signal = shift_signal
        return signal


class ShiftUD(object):
    def __init__(self, p=0.2, shift_p=[0.01, 0.05]):
        self.p = p
        self.shift_p = shift_p

    def __call__(self, signal):
        if self.p > numpy.random.rand():
            max_value = numpy.max(signal)
            target_shift_p = numpy.random.uniform(self.shift_p[0], self.shift_p[1])
            offset_value = max_value * target_shift_p
            if numpy.random.rand() > 0.5:
                signal += offset_value
            else:
                signal -= offset_value
        return signal


class Normalizer(object):
    def __init__(self, with_std=False, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = numpy.array([[mean]])
        self.std = numpy.array([[std]])
        self.with_std = with_std

    def __call__(self, signal):
        max = numpy.max(signal, axis=1)
        min = numpy.min(signal, axis=1)
        if self.with_std:
            signal = (((signal.astype(numpy.float32) - min) / (max - min)) - self.mean) / self.std
        else:
            signal = ((signal.astype(numpy.float32) - min) / (max - min))

        return signal


class Resizer(object):
    def __init__(self, signal_size=1024):
        self.signal_size = signal_size

    def __call__(self, signal):
        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)

        return signal
