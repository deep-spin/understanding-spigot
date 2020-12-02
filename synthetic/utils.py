"""
This code is from Goncalo Correia.
"""

class RunningStats:

    def __init__(self, update_factor=0.05):
        self.update_factor = update_factor
        self.avg = 0.
        self.std = 1.

    def update(self, avg, std):
        self.avg = self.update_factor * avg + \
            (1 - self.update_factor) * self.avg
        self.std = max(
            1.,
            self.update_factor * std + (1 - self.update_factor) * self.std)
