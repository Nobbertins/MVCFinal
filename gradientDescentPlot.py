#import necessary libraries
from math import sqrt, sin, cos, exp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

#specify the function
def f(x, y):
    return pow(x,2) + pow(y,2)

#specify inputs
steps = 100
learningRate = 0.2
startPoint = [0, 0]
displaySkips = 5

def findNegativeGradient(x, y):
    #It's important to note that the gradient can be pre-defined for a differentiable function (for f(x,y) = x^2 + y^2 it would be <2x, 2y>)
    #But estimating this way is more practical since each function has its own gradient

    #use the difference quotient with a very small h value to estimate partial derivatives 
    h = 0.0001
    dx = (f(x + h, y) - f(x, y))/h
    dy = (f(x, y + h) - f(x, y))/h
    #return the negative gradient since we want to descend not ascend
    return [-dx, -dy, -1]

def findMinimum(startPoint, steps, learningRate, displaySkips):
    x, y = startPoint
    data = [((x, y, f(x, y)), (0,0,0))]
    #descend the specified number of steps:
    for step in range(steps):
        #output the point before each step (only do this if the step is < 1000 otherwise that will be a lot of print statements!)
        #print(f"Step {step}: ({x}, {y}, {f(x,y)})")

        #find the negative gradient for the current point
        grad = findNegativeGradient(x, y)
        #calculate the vector's magnitude
        magnitude = sqrt(pow(grad[0],2) + pow(grad[1],2) + pow(grad[2],2))
        #normalize the gradient vector into a unit vector (for it to specify direction)
        norm = (grad[0]/magnitude, grad[1]/magnitude, grad[2]/magnitude)
        #add the direction * learningRate to the final point
        x += norm[0] * learningRate
        y += norm[1] * learningRate
        point = (x, y, f(x, y))
        data.append((point, norm))
    #after the descent return the point (which should hopefully be a minimum)
    return (point, data[::displaySkips])

#graph the surface
xData = np.linspace(-6, 6, 1000)
yData = np.linspace(-6, 6, 1000)

# Initialize an empty array for z values
zData = np.zeros((len(xData), len(yData)))

# Compute z values using the function f
for i in range(len(xData)):
    for j in range(len(yData)):
        zData[i, j] = f(xData[i], yData[j])

xData, yData = np.meshgrid(xData, yData)

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surf = ax.plot_surface(xData, yData, zData, cmap='viridis', alpha = 0.4)

# Customize the plot
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('Gradient Descent')

# Add a color bar
fig.colorbar(surf)

# Add a slider for the timeline
ax_timeline = plt.axes([0.2, 0.06, 0.6, 0.03], facecolor='lightgoldenrodyellow')
ay_timeline = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor='lightgoldenrodyellow')
x_timeline = Slider(ax_timeline, 'X', -6, 6, valinit=0, valstep=0.1)
y_timeline = Slider(ay_timeline, 'Y', -6, 6, valinit=0, valstep=0.1)

def updateDescentPath(start):
    ax.clear()
    # Plot the 3D surface
    surf = ax.plot_surface(xData, yData, zData, cmap='viridis', alpha = 0.4)
    min, data = findMinimum(start, steps, learningRate, displaySkips)
    for pt, norm in data:
        x, y, z = pt
        u, v, w = norm
        ax.scatter(x, y, z, color='red', s=10, zorder = 5, alpha = 1)
        ax.quiver(x, y, z, u, v, w, color='red')

# Define the update function for the slider
def updateX(val):
    startPoint[0] = val
    updateDescentPath(startPoint)
def updateY(val):
    startPoint[1] = val
    updateDescentPath(startPoint)

x_timeline.on_changed(updateX)
y_timeline.on_changed(updateY)


#Plot the start point
# startX, startY, startZ = data[0][0]
# pt = ax.scatter(startX, startY, startZ, color='red', s=50, zorder = 5)

# Show plot
plt.show()