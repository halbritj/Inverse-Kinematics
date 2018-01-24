import numpy as np
import time
import matplotlib.pyplot as plt

class plotter:
    def __init__(self):
        self.fig = plt.figure()
        self.graphs = []
        for i in range(6):
            temp = plt.subplot('61%d' %(i))
            plot, = temp.plot([], [])
            self.graphs.append(plot)

    def add(self, data):
        for graph, point in zip(self.graphs, data):
            #print(graph, point)
            graph.set_xdata( np.append(graph.get_xdata(), [len(graph.get_xdata())]) )
            graph.set_ydata( np.append(graph.get_ydata(), [point]) )
            graph.axes.relim()
            graph.axes.autoscale_view()
        plt.pause(.01)
        plt.draw()
        
'''
p = plotter()


for i in range(200):
    a = np.random.rand(6)
    p.add(a)
'''

