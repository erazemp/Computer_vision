import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from methods import lucas_kanade, horn_schunck, horn_schunck_improved, lucas_kanade_improved

from ex1_utils import rotate_image, show_flow
import time


def flow():
    im1 = np.random.rand(200, 200).astype(np.float32)
    im2 = im1.copy()
    im2 = rotate_image(im2, -1)

    start = time.time()
    U_lk, V_lk = lucas_kanade(im1, im2, 3)
    print("Lucas-Kanade runtime: {:.5f} seconds".format(time.time() - start))

    #print(U_lk)
    #print(V_lk)
    start = time.time()
    U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
    print("HSruntime: {:.5f} seconds".format(time.time() - start))


    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow')

    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type='angle')
    show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    fig2.suptitle('Horn−Schunck Optical Flow ')
    plt.show()


def selected_image_flow():
    file_path = os.path.dirname(__file__)
    f_path = os.path.join(file_path, "disparity")


    # cporta_left.png
    # cporta_right.png
    # office2_left.png
    # office2_right.png
    im1 = os.path.join(f_path, 'office2_left.png')
    im2 = os.path.join(f_path, 'office2_right.png')
    im1 = cv2.imread(im1, cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread(im2, cv2.IMREAD_GRAYSCALE)
    #im1 = cv2.imread(im1,cv2.IMREAD_COLOR)
    #im2 = cv2.imread(im2, cv2.IMREAD_COLOR)

    start = time.time()
    U_lk, V_lk = lucas_kanade(im1, im2, 3)
    print("Lucas-Kanade runtime: {:.5f} seconds".format(time.time() - start))
    n = 3


    U_lk_im, V_lk_im = lucas_kanade_improved(im1, im2, n)

    start = time.time()
    U_hs_im, V_hs_im = horn_schunck_improved(im1, im2, 1000, 0.5, n)
    print("H-S with L-C runtime: {:.5f} seconds".format(time.time() - start))

    U_hs01, V_hs02 = horn_schunck(im1, im2, 1000, 0.5)
    im1 = im1 / 255
    im2 = im2 / 255
    start = time.time()
    U_lk_norm, V_lk_norm = lucas_kanade_improved(im1, im2, n)
    print("Lucas-Kanade improved runtime: {:.5f} seconds".format(time.time() - start))
    U_lk_im_norm, V_lk_im_norm = lucas_kanade(im1, im2, n)

    #U_hs_im, V_hs_im = horn_schunck_improved(im1, im2, 1000, 0.5, 3)

    start = time.time()
    U_hs, V_hs = horn_schunck(im1, im2, 1000, 0.5)
    print("H-S runtime: {:.5f} seconds".format(time.time() - start))




    fig1, ((ax1_11, ax1_12), (ax1_21, ax1_22)) = plt.subplots(2, 2)
    ax1_11.imshow(im1)
    ax1_12.imshow(im2)
    show_flow(U_lk, V_lk, ax1_21, type='angle')
    show_flow(U_lk, V_lk, ax1_22, type='field', set_aspect=True)
    fig1.suptitle('Lucas−Kanade Optical Flow no norm')

    fig5, ((ax5_11, ax5_12), (ax5_21, ax5_22)) = plt.subplots(2, 2)
    ax5_11.imshow(im1)
    ax5_12.imshow(im2)
    show_flow(U_hs01, V_hs02, ax5_21, type='angle')
    show_flow(U_hs01, V_hs02, ax5_22, type='field', set_aspect=True)
    fig5.suptitle('Horn-Schunck Optical Flow no norm')

    ## horn-schunck

    fig2, ((ax2_11, ax2_12), (ax2_21, ax2_22)) = plt.subplots(2, 2)
    ax2_11.imshow(im1)
    ax2_12.imshow(im2)
    show_flow(U_hs, V_hs, ax2_21, type='angle')
    show_flow(U_hs, V_hs, ax2_22, type='field', set_aspect=True)
    fig2.suptitle('Horn−Schunck Optical Flow + norm ')

    fig3, ((ax3_11, ax3_12), (ax3_21, ax3_22)) = plt.subplots(2, 2)
    ax3_11.imshow(im1)
    ax3_12.imshow(im2)
    show_flow(U_hs_im, V_hs_im, ax3_21, type='angle')
    show_flow(U_hs_im, V_hs_im, ax3_22, type='field', set_aspect=True)
    fig3.suptitle('Horn−Schunck with lucas-Kanade output Optical Flow ')

    fig4, ((ax4_11, ax4_12), (ax4_21, ax4_22)) = plt.subplots(2, 2)
    ax4_11.imshow(im1)
    ax4_12.imshow(im2)
    show_flow(U_lk_im, V_lk_im, ax4_21, type='angle')
    show_flow(U_lk_im, V_lk_im, ax4_22, type='field', set_aspect=True)
    fig4.suptitle('Improved Lucas-kanade ')

    fig6, ((ax6_11, ax6_12), (ax6_21, ax6_22)) = plt.subplots(2, 2)
    ax6_11.imshow(im1)
    ax6_12.imshow(im2)
    show_flow(U_lk_norm, V_lk_norm, ax6_21, type='angle')
    show_flow(U_lk_norm, V_lk_norm, ax6_22, type='field', set_aspect=True)
    fig6.suptitle('Improved Lucas-kanade + norm ')


    fig7, ((ax7_11, ax7_12), (ax7_21, ax7_22)) = plt.subplots(2, 2)
    ax7_11.imshow(im1)
    ax7_12.imshow(im2)
    show_flow(U_lk_im_norm, V_lk_im_norm, ax7_21, type='angle')
    show_flow(U_lk_im_norm, V_lk_im_norm, ax7_22, type='field', set_aspect=True)
    fig7.suptitle('Lucas-kanade + norm ')
    plt.show()





if __name__ == "__main__":
    flow()
    #selected_image_flow()

