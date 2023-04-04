import math

import numpy as np

from ex2_utils import *
import mean_shift_methods

def get_matrices(dims, size):
    sh = int(dims[1] / 2)
    sh2 = int(dims[0] / 2)
    x = np.linspace(-sh, sh, int(size[1]))
    y = np.linspace(-sh2, sh2, int(size[0]))
    xi, yi = np.meshgrid(x, y)

    return xi, yi


def calculate_grid(xi, yi, wi):
    xk = np.divide(np.sum(np.multiply(xi, wi)), np.sum(wi))
    yk = np.sum(np.multiply(yi, wi))
    yk = yk / np.sum(wi)

    return xk, yk


class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        #self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        self.bins = 8
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])
        self.kernel = create_epanechnik_kernel(region[2], region[3], self.parameters.sigma)
        patch, _ = get_patch(image, self.position, np.shape(self.kernel))
        self.hist_q = extract_histogram(patch, nbins=self.bins, weights=self.kernel)
        self.hist_q = (self.hist_q / np.sum(self.hist_q))

    def track(self, image):
        # alfa 0 , 8bins, thershold,
        self.shape = np.shape(self.kernel)
        tab_r = self.seeking(image, self.shape)

        #return shift_x, shift_y, size1, size2
        return [tab_r[0], tab_r[1], tab_r[2], tab_r[3]]

    def seeking(self, img, ker):
        threshold = 0.01
        patch, inliers = get_patch(img, self.position, (ker[1], ker[0]))
        hist_p = extract_histogram(patch, nbins=self.bins, weights=(self.kernel * inliers))
        # normalize hist
        hist_p = (hist_p / np.sum(hist_p))
        v = np.sqrt(self.hist_q / (hist_p + 0.00001))
        wi = backproject_histogram(patch, v, nbins=self.bins)
        dimensions = np.shape(wi)

        sh = math.floor(dimensions[1] / 2)
        sh2 = math.floor(dimensions[0] / 2)
        #xi, yi = np.meshgrid(np.arange(-sh, sh + 1), np.arange(-sh2, sh2 + 1))
        xi, yi = get_matrices(dimensions, ker)


        for i in range(50):
            patch, inliers = get_patch(img, self.position, (ker[1], ker[0]))
            hist_p = extract_histogram(patch, nbins=self.bins, weights=(self.kernel * inliers))
            #normalize hist
            hist_p = (hist_p / np.sum(hist_p))
            v = np.sqrt(self.hist_q / (hist_p + 0.00001))
            wi = backproject_histogram(patch, v, nbins=self.bins)
            #dimensions = np.shape(wi)


            xk, yk = calculate_grid(xi, yi, wi)

            # recenter:
            center_x = self.position[0] + xk
            center_y = self.position[1] + yk
            self.position = [center_x, center_y]

           # patch, _ = get_patch(img, self.position, self.kernel)
           # q = extract_histogram(patch, self.bins, self.kernel)
           # self.hist_q = np.add(((1 - 0.5) * self.hist_q), (0.5 * q))
            #self.hist_q = (self.hist_q / np.sum(self.hist_q))


            #thresh
            if np.abs(xk) < threshold and np.abs(yk) < threshold:
                print(i)
                if i >= 20:
                    print("iterations: ", i)
                break

        left = max(round(self.position[0] - float(self.size[0]) / 2), 0)
        top = max(round(self.position[1] - float(self.size[1]) / 2), 0)

        #left = self.position[0] - self.size[0] / 2
        #top = self.position[1] - self.size[1] / 2

        return [left, top, self.size[0], self.size[1]]




class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        self.sigma = 0.6
        self.epsilon = 0.5

