import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure()
ax = plt.axes()
ax.quiver(0,0,1,1,color=(1, 0, 0, 0.3),angles='xy', scale_units='xy', scale=0.5)
ax.quiver([0,0],[0,0],[1,0],[0,1],color=[(1, 0, 0, 0.3), (0, 1, 0, 0.3)],angles='xy', scale_units='xy', scale=1)
ax.grid()
ax.set_xlabel('X')
ax.set_xlim(-1, 1)
ax.set_ylabel('Y')
ax.set_ylim(-1, 1)
plt.show()
plt.pause(0)
plt.clf()  #清除图像