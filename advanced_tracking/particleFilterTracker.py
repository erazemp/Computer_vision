from utils.tracker import Tracker
import numpy as np


class ParticleFilterTracker(Tracker):

    def name(self):
        return 'ParticleFilterTracker'

    def new_particles(self, weights, particles, N):
        weights_norm = weights / np.sum(weights)
        weights_sum = np.cumsum(weights_norm)
        random_samples = np.random.rand(N, 1)
        sampled_idx = np.digitize(random_samples, weights_sum)
        new_particles = particles[sampled_idx.flatten(), :]

        return new_particles
