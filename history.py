import random
import collections


class ReplayBuffer:
    def __init__(self, max_len=512):
        self._max_len = max_len
        self._buffer = collections.deque()

    def insert(self, *args):
        if len(self._buffer) >= self._max_len:
            self._buffer.popleft()
        self._buffer.append(*args)

    def get_batch(self, n_batch, n_lost):
        # forget random items
        for _ in range(n_lost):
            del self._buffer[random.randrange(len(self._buffer))]
        # gather a mini batch
        batch = random.sample(self._buffer, n_batch)
        return zip(*batch)

    def reset(self):
        self._buffer.clear()

    def len(self):
        return len(self._buffer)

    def is_full(self):
        return len(self._buffer) == self._max_len
