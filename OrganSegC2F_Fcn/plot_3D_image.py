import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys


class Counter:
    def __init__(self, n, init=0):
        assert init < n
        self.counter = init
        self.max = n

    def up(self):
        self.counter = (self.counter - 1) % self.max

    def down(self):
        self.counter = (self.counter + 1) % self.max


class Multi3DArrayPlane:

    def __init__(self, figure, rows_sub_figure, columns_sub_figure):
        self.figure = figure
        self.rows = rows_sub_figure
        self.columns = columns_sub_figure
        self.ax_list = []
        self.array_list = []
        self.imshow_param_list = []
        self.need_draw = False
        self.index_list = []

    def add(self, array, title = None, vmin=None, vmax=None, cmap='nipy_spectral', fixed_window=True):
        if fixed_window:
            if vmin is None:
                vmin = array.min()
            if vmax is None:
                vmax = array.max()
        self.index_list.append(Counter(array.shape[0], int(array.shape[0] / 2.0)))
        ax = self.figure.add_subplot(self.rows, self.columns, len(self.ax_list) + 1)
        if title is not None:
            ax.set_title(title)
        self.ax_list.append(ax)
        self.array_list.append(array)
        self.imshow_param_list.append((vmin, vmax, cm.get_cmap(cmap)))

    def _refresh(self):
        for i in range(len(self.ax_list)):
            image = self.array_list[i][self.index_list[i].counter]
            vmin = self.imshow_param_list[i][0]
            vmax = self.imshow_param_list[i][1]
            if vmin is None:
                vmin = np.min(image)
            if vmax is None:
                vmax = np.max(image)
            ax = self.ax_list[i]
            ax.set_ylabel('slice {}'.format(self.index_list[i].counter))
            axes_image=ax.get_images()[0]
            axes_image.set_data(image)
            axes_image.set_clim(vmin, vmax)

        self.figure.canvas.draw_idle()

    def ready(self):
        def scroll_fun(event):
            if event.button == 'up':
                for index in self.index_list:
                    index.up()
            elif event.button == 'down':
                for index in self.index_list:
                    index.down()
            else:
                print(event.button)
            self._refresh()
        self.figure.canvas.mpl_connect('scroll_event', scroll_fun)

        for i in range(len(self.ax_list)):
            image = self.array_list[i][self.index_list[i].counter]
            vmin = self.imshow_param_list[i][0]
            vmax = self.imshow_param_list[i][1]
            if vmin is None:
                vmin = np.min(image)
            if vmax is None:
                vmax = np.max(image)
            cmap = self.imshow_param_list[i][2]
            ax = self.ax_list[i]
            ax.set_ylabel('slice {}'.format(self.index_list[i].counter))
            self.figure.colorbar(ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, animated=True), ax=ax)
        self.figure.canvas.draw()

if __name__ == '__main__':
    array_list = []
    npy_file_list = ["/media/jionie/Disk1/images/0031.npy", "/media/jionie/Disk1/labels/0031.npy"]
    print(npy_file_list)
    for npy_file in npy_file_list:
        print('loading:', npy_file)
        a = np.load(npy_file)
        array_list.append(a)
        print(a.min(), a.max())
    fig = plt.figure()
    rows = np.floor(np.sqrt(len(array_list)))
    if rows == 0:
        rows = 1
    columns = np.ceil(len(array_list) / float(rows))
    plane = Multi3DArrayPlane(fig, rows, columns)
    for a in array_list:
        plane.add(a.transpose((2, 0, 1))) # 201-Z, 012-X, 102-Y
    plane.ready()
    plt.show()