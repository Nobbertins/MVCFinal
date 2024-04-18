from math import sqrt, sin, cos, exp
from random import random
#specify the function
def f(x, y):
    #return pow(x,2) + pow(y,2)
    
    #non-trivial case (non-uniform valleys and peaks)
    return 2*sin(x) + 1*cos(y) - 8*exp(-(pow(x-2, 2) + pow(y-1,2))) + 0.1*(pow(x,2)+pow(y,2))

#specify inputs
steps = 100
learningRate = 0.1

#num of random samples
samples = 100
#define range
xMin, xMax = (-6, 6)
yMin, yMax = (-6, 6)

#calculate negative gradient of the function at a point
def findNegativeGradient(x, y):
    #It's important to note that the gradient can be pre-defined for a differentiable function (for f(x,y) = x^2 + y^2 it would be <2x, 2y>)
    #But estimating this way is more practical since each function has its own gradient

    #use the difference quotient with a very small h value to estimate partial derivatives 
    h = 0.000001
    dx = (f(x + h, y) - f(x, y))/h
    dy = (f(x, y + h) - f(x, y))/h
    #return the negative gradient since we want to descend not ascend
    return [-dx, -dy]

#find the minimum using gradient descent for a single point
def findMinimum(startPoint):
    x, y = startPoint
    #descend the specified number of steps:
    for step in range(steps):
        #output the point before each step (only do this if the step is < 1000 otherwise that will be a lot of print statements!)
        #print(f"Step {step}: ({x}, {y}, {f(x,y)})")

        #find the negative gradient for the current point
        grad = findNegativeGradient(x, y)
        #calculate the vector's magnitude
        magnitude = sqrt(pow(grad[0],2) + pow(grad[1],2))
        #normalize the gradient vector into a unit vector (for it to specify direction)
        norm = [grad[0]/magnitude, grad[1]/magnitude]
        #add the direction * learningRate to the final point
        x += norm[0] * learningRate
        y += norm[1] * learningRate
    #after the descent return the point (which should hopefully be a minimum)
    return (x, y, f(x,y))

#find the minimum of all samples and return the lowest one
def findMinimumWithSamples(samples):
    #variables to store "absolute" minimum values
    lowestZ = None
    lowestMin = None

    #use gradient descent on each sample
    for i in range(samples):
        #random start point in given range
        startPoint = [random()*(xMax-xMin)+xMin, random()*(yMax-yMin)+yMin]
        #run gradient descent and find minimum
        min = findMinimum(startPoint)
        x, y, z = min

        #output the result of each random sample (only do this if samples < 100 otherwise that will be a lot of print statements!)
        #print(f"Local Minimum Found: ({x}, {y}, {z})")

        #first minimum is the lowest one found
        if i == 0:
            lowestZ = z
            lowestMin = min
        #if z value is lower, then this min is the new lowest
        if z < lowestZ:
            lowestZ = z
            lowestMin = min
            
    return lowestMin

#print output of random sampling function (the lowest min found)
x, y, z = findMinimumWithSamples(samples)
print(f"The Lowest Minimum Found is ({x}, {y}, {z})")