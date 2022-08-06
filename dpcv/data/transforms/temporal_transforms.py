"""
code modified form https://github.com/kenshohara/3D-ResNets-PyTorch.git
"""
import random
import math
import numpy as np


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalBeginCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalTwoEndsCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        from_start = frame_indices[:self.size]
        to_end = frame_indices[-self.size:]

        return [from_start, to_end]


class TemporalCenterCrop(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):

    def __init__(self, size):
        self.size = size
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        if len(out) < self.size:
            out = self.loop(out)

        return out


class TemporalDownsample:

    def __init__(self, length=100):
        self.len = length

    def __call__(self, frame_indices):
        sample_frames = np.linspace(0, len(frame_indices), self.len, endpoint=False, dtype=np.int32)
        frame_indices = np.array(frame_indices)[sample_frames]
        return frame_indices.tolist()


class TemporalEvenCropDownsample:

    def __init__(self, segment_len=16, num_segmets=6):
        self.segment_len = segment_len
        self.num_segments = num_segmets

    def __call__(self, frame_indices):

        assert len(frame_indices) > self.num_segments * self.segment_len, \
            "index out of range, segment_len * num_segments must less then the number of frame_indices"
        segments = []
        for i in range(self.num_segments):
            start_index, end_index = i * self.segment_len, (i + 1) * self.segment_len
            segments.append([frame_indices[idx] for idx in range(start_index, end_index)])

        return segments


class TemporalEvenCrop(object):

    def __init__(self, size, n_samples=1):
        self.size = size
        self.n_samples = n_samples
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        n_frames = len(frame_indices)
        stride = max(
            1, math.ceil((n_frames - 1 - self.size) / (self.n_samples - 1)))

        out = []
        for begin_index in frame_indices[::stride]:
            if len(out) >= self.n_samples:
                break
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


class SlidingWindow(object):

    def __init__(self, size, stride=0):
        self.size = size
        if stride == 0:
            self.stride = self.size
        else:
            self.stride = stride
        self.loop = LoopPadding(size)

    def __call__(self, frame_indices):
        out = []
        for begin_index in frame_indices[::self.stride]:
            end_index = min(frame_indices[-1] + 1, begin_index + self.size)
            sample = list(range(begin_index, end_index))

            if len(sample) < self.size:
                out.append(self.loop(sample))
                break
            else:
                out.append(sample)

        return out


class TemporalSubsampling(object):

    def __init__(self, stride):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]


class Shuffle(object):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, frame_indices):
        frame_indices = [
            frame_indices[i:(i + self.block_size)]
            for i in range(0, len(frame_indices), self.block_size)
        ]
        random.shuffle(frame_indices)
        frame_indices = [t for block in frame_indices for t in block]
        return frame_indices
