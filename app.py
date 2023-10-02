import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# ----------------------------------------------------
# Obtaining data
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])

datapoints = []

def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    ax.scatter(event.xdata, event.ydata)
    datapoints.append(str(event.xdata) + "," + str(event.ydata) + "\n")
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

# Escribimos datos obtenidos en test.csv
file = open('data.csv', 'w+')
file.write("studytime,score\n")
file.writelines(datapoints)
file.close()
# ----------------------------------------------------

# Leemos los datos
points = pd.read_csv('data.csv')

# Funcion de error cuadrado
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = float(len(points))
    for i in range(len(points)):
        x = points.iloc[i, 0]
        y = points.iloc[i, 1]
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return [m, b]

m = 0
b = 0
L = 0.001
epochs = 1000

min_x = 2
max_x = 10

for i in range(epochs):
    m, b = gradient_descent(m, b, points, L)

print(m, b)

plt.scatter(points.iloc[:,0], points.iloc[:,1], color='pink')
plt.plot(list(range(min_x, max_x)), [m * x + b for x in range(min_x, max_x)], color='blue')
plt.show()