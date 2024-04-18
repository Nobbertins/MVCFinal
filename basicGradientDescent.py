from math import sqrt

#specify the function
def f(x, y):
    return pow(x,2) + pow(y,2)

#specify inputs
steps = 100000
learningRate = 0.1
startPoint = [640.32, 832.27]

def findNegativeGradient(x, y):
    #It's important to note that the gradient can be pre-defined for a differentiable function (for f(x,y) = x^2 + y^2 it would be <2x, 2y>)
    #But estimating this way is more practical since each function has its own gradient

    #use the difference quotient with a very small h value to estimate partial derivatives 
    h = 0.0001
    dx = (f(x + h, y) - f(x, y))/h
    dy = (f(x, y + h) - f(x, y))/h
    #return the negative gradient since we want to descend not ascend
    return [-dx, -dy]

def findMinimum(startPoint):
    x, y = startPoint
    #descend the specified number of steps:
    for step in range(steps):
        #the following line outputs the point before each step (only do this if the step is < 1000 otherwise that will be a lot of print statements!)
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

min = findMinimum(startPoint)
x, y, z = min
print(f"Local Minimum Found: ({x}, {y}, {z})")
