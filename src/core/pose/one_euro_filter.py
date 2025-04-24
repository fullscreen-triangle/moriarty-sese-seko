import math
import ray


@ray.remote
class DistributedOneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_previous = None
        self.dx = None

    @ray.remote
    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq
        
        dx_smoothed = self.filter_dx(self.dx, self.get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, self.get_alpha(self.freq, cutoff))
        
        self.x_previous = x
        return x_filtered

    @staticmethod
    @ray.remote
    def get_alpha(rate=30, cutoff=1):
        tau = 1 / (2 * math.pi * cutoff)
        te = 1 / rate
        return 1 / (1 + tau / te)

# Keep original class for compatibility
OneEuroFilter = DistributedOneEuroFilter


if __name__ == '__main__':
    filter = OneEuroFilter(freq=15, beta=0.1)
    for val in range(10):
        x = val + (-1)**(val % 2)
        x_filtered = filter(x)
        print(x_filtered, x)
