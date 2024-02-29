import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np

# Assuming 'im' is your image
im = np.random.rand(300, 500)

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

# Initial slice
min_index = 0
max_index = 100
l = plt.imshow(im[min_index:max_index, 100:400])

ax_min = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_max = plt.axes([0.25, 0.15, 0.65, 0.03])

s_min = widgets.Slider(ax_min, 'Min', 0, im.shape[0]-1, valinit=min_index)
s_max = widgets.Slider(ax_max, 'Max', 0, im.shape[0]-1, valinit=max_index)

def update(val):
    min_index = int(s_min.val)
    max_index = int(s_max.val)
    l.set_data(im[min_index:max_index, 100:400])
    fig.canvas.draw_idle()

s_min.on_changed(update)
s_max.on_changed(update)

plt.show()