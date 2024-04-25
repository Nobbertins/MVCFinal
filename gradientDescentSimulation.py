#import necessary libraries
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider

import matplotlib.animation as animation


#specify the function
def f(x, y):
    #paraboloid
    return pow(x, 2) + pow(y, 2)

#specify inputs
steps = 500
learningRate = 0.1
startPoint = [3.5, 5.5]
displaySkips = 1

#function to find negative gradient (used to find direction of descent)
def findNegativeGradient(x, y):
    #It's important to note that the gradient can be pre-defined for a differentiable function (for f(x,y) = x^2 + y^2 it would be <2x, 2y>)
    #But estimating this way is more practical since each function has its own gradient

    #use the difference quotient with a very small h value to estimate partial derivatives 
    h = 0.0001
    dx = (f(x + h, y) - f(x, y))/h
    dy = (f(x, y + h) - f(x, y))/h
    #return the negative gradient since we want to descend not ascend
    return [-dx, -dy, -1]

#function to find local minimum using gradient descent from a starting point
def findMinimum(startPoint, steps, learningRate, displaySkips):
    #initialize data array to store points
    x, y = startPoint
    data = [(x, y, f(x, y))]
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
        #add the point to the data array
        data.append(point)
    #after the descent return the point array (which should hopefully be a minimum) to use for the animation skipping a set amount of frames
    return data[::displaySkips]

#define bounds for the surface
xData = np.linspace(-6, 6, 1000)
yData = np.linspace(-6, 6, 1000)

# Initialize an empty array for z values
zData = np.zeros((len(xData), len(yData)))

# Compute z values using the function f
for i in range(len(xData)):
    for j in range(len(yData)):
        zData[i, j] = f(xData[i], yData[j])

#reshape data to be graphed
xData, yData = np.meshgrid(xData, yData)

# Create a figure and axis
fig = plt.figure()

# Creating legend with color box
start_patch = mpatches.Patch(color='green', label='Start Point') 
moving_patch = mpatches.Patch(color='black', label='Current Point') 
end_patch = mpatches.Patch(color='red', label='End Point') 
plt.legend(handles=[start_patch, moving_patch, end_patch], loc = 'upper left')

#create 3D axes
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surf = ax.plot_surface(xData, yData, zData, cmap='viridis', alpha = 0.4)

# Label the plot
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
ax.set_title('Gradient Descent Simulation')

# Add a color bar
fig.colorbar(surf)

# Add space for the sliders to go in
ax_timeline = plt.axes([0.2, 0.06, 0.6, 0.03], facecolor='lightgoldenrodyellow')
ay_timeline = plt.axes([0.2, 0.04, 0.6, 0.03], facecolor='lightgoldenrodyellow')
aL_timeline = plt.axes([0.2, 0.02, 0.6, 0.03], facecolor='lightgoldenrodyellow')
aS_timeline = plt.axes([0.2, 0, 0.6, 0.03], facecolor='lightgoldenrodyellow')

# Add sliders in these spaces
x_timeline = Slider(ax_timeline, 'Start X', -6, 6, valinit= 3.5, valstep= 0.1)
y_timeline = Slider(ay_timeline, 'Start Y', -6, 6, valinit= 5.5, valstep= 0.1)
lr_timeline = Slider(aL_timeline, 'Learning Rate', 0.1, 1, valinit = 0.1, valstep = 0.01)
s_timeline = Slider(aS_timeline, 'Steps', 100, 500, valinit= 1, valstep= 1)

#define initial points
data = findMinimum(startPoint, steps, learningRate, displaySkips)
x, y, z = data[0]
a, b, c = data[-1]
#graph start point
ax.scatter(x, y, z, color='green', s = 15, zorder = 6, alpha = 1)
#graph end point
ax.scatter(a, b, c, color='red', s = 15, zorder = 6, alpha = 1)

#define list of points to be changed per input update
scat = ax.scatter(0, 0, 0, color='black', s= 15, zorder = 5, alpha = 1)

#define empty list of data to be changed per input update
xL = np.zeros(len(data))
yL = np.zeros(len(data))
zL = np.zeros(len(data))

#fill list with initial descent path
for i in range(len(data)):
    xL[i] = data[i][0]
    yL[i] = data[i][1]
    zL[i] = data[i][2]

#define update function for the animation
def update(frame):
    # for each frame, display the next point in the descent
    x = np.array([xL[frame]])
    y = np.array([yL[frame]])
    z = np.array([zL[frame]])
    # update the scatter plot
    scat._offsets3d = (x, y, z)
    #return a single element tuple of the scatter plot for the animation
    return scat,

#function to update the descent path points per input change
def updateDescentPath(start):
    #clear and relabel axes
    ax.clear()
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_zlabel('Z-Axis')
    ax.set_title('Gradient Descent Simulation')

    #re-plot 3D surface
    ax.plot_surface(xData, yData, zData, cmap='viridis', alpha = 0.4)

    #update data by rerunning the gradient descent algorithm with new inputs
    global scat
    data = findMinimum(start, steps, learningRate, displaySkips)
    x, y, z = data[0]
    #plot start point
    ax.scatter(x, y, z, color='green', s = 15, zorder = 6, alpha = 1)
    #plot end point
    ax.scatter(a, b, c, color='red', s = 15, zorder = 6, alpha = 1)
    #redefine the plotted point list
    scat = ax.scatter(0, 0, 0, color='black', s= 15, zorder = 5, alpha = 1)
    #redefine point list
    for i in range(len(data)):
        xL[i] = data[i][0]
        yL[i] = data[i][1]
        zL[i] = data[i][2]

# Define the update functions for the sliders
def updateX(val):
    startPoint[0] = val
    updateDescentPath(startPoint)
def updateY(val):
    startPoint[1] = val
    updateDescentPath(startPoint)
def updateSteps(val):
    global steps
    steps = val
    updateDescentPath(startPoint)
def updateLearningRate(val):
    global learningRate
    learningRate = val
    updateDescentPath(startPoint)

# Call update functions on changes in the sliders
x_timeline.on_changed(updateX)
y_timeline.on_changed(updateY)
lr_timeline.on_changed(updateLearningRate)
s_timeline.on_changed(updateSteps)

#define the animation manager
ani = animation.FuncAnimation(fig=fig, func=update, frames = 100, interval= 10)
# Show plot
plt.show()