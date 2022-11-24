import math
from matplotlib import pyplot as plt
import numpy as np
while 1:
    # plt.cla()
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.imshow(np.random.randint(0, 2, (2, 2)))
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.imshow(np.random.randint(0, 2, (2, 2)))
    # plt.pause(0.1)
    # ax1.cla()
    # ax2.cla()

    plt.ion()  #打开交互模式
    # ax1 = plt.subplot(1, 2, 1)
    # ax1.imshow(np.random.randint(0, 2, (2, 2)))
    # ax2 = plt.subplot(1, 2, 2)
    # ax2.imshow(np.random.randint(0, 2, (2, 2)))
    ax = plt.axes()
    ax.quiver(0,0,np.random.rand(1)*2-1,np.random.rand(1)*2-1,color=[(1, 0, 0, 0.3), (0, 1, 0, 0.3)],angles='xy', scale_units='xy', scale=1)
    ax.grid()
    ax.set_xlabel('X')
    ax.set_xlim(-1, 1)
    ax.set_ylabel('Y')
    ax.set_ylim(-1, 1)
    plt.show()
    plt.pause(1/100)
    plt.clf()  #清除图像