import math

from ex2_utils import *
import numpy as np
import random


def find_max(arr):
    max_x = np.where(arr == np.amax(arr))[0][0]
    max_y = np.where(arr == np.amax(arr))[1][0]
    #max_val = np.amax(arr)
    return max_x, max_y

def get_matrices(k_size):
    #xi = np.array([[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]])
    #yi = np.array([[-2, -2, -2, -2, -2], [-1, -1, -1, -1, -1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

    #x = np.linspace(-int(k_size[0] / 2), int(k_size[0] / 2), k_size[0])
    #y = np.linspace(-int(k_size[1] / 2), int(k_size[1] / 2), k_size[1])


    sf1 = math.floor(k_size[0] / 2)
    sf2 = math.floor(k_size[1] / 2)
    x = np.linspace(-sf1, sf1, int(k_size[0]))
    y = np.linspace(-sf2, sf2, int(k_size[1]))

    xi, yi = np.meshgrid(x, y)

    return xi, yi


def mean_shift_seeking(kernel_size):
    img = generate_responses_2()
    thresh = 0.008
    ### replace all 0 in img:
    img[img == 0] = 2.22e-16
    xi, yi = get_matrices(kernel_size)
    # center
    center = [random.randint(30, 60), random.randint(30, 60)]
    print("center: ", center)
    #center = [32, 32]
    max_t = find_max(img)
    print("maximum of the genrated img: ",max_t )

    nmb_of_its = 10000
    count = 0
    while count < nmb_of_its:
        count += 1
        wi, _ = get_patch(img, center, kernel_size)
        wi_sum = np.sum(wi)
       # xk = np.sum(np.multiply(xi, wi))
        xk = np.divide(np.sum(np.multiply(xi, wi)), np.sum(wi))
        yk = np.sum(np.multiply(yi, wi))
        # divide
        #xk = xk / wi_sum
        yk = yk / wi_sum

        # recenter:
        center_x = center[0] + xk
        center_y = center[1] + yk
        # center = [[0] + xk, [1] + yk]
        center = [center_x, center_y]

        if xk < thresh and yk < thresh:
            break

    print("number of iterations: ", count)
    print("end location: ", center)
    return int(center[0]), int(center[1])


if __name__ == '__main__':
    x, y = mean_shift_seeking((5, 5))
