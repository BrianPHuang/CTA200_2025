import numpy as np

def lorenz_equation(t, W, sigma, r, b):
    """
    Compute the time derivative for the lorenz equation to get convection equation.
    
    Parameters:
    -----------
    t: float, time variable, which is required by ODE system.
    N: list or numpy array, which contains the current value of variables [X, Y, Z].
    sigma: float, Prandlt number.
    r: float, Rayleigh number.
    b: float, dimensionless length scale.
    
    Returns:
    -----------
    list that contains the time derivative of the variable:
    [dX/dt, dY/dt, dZ/dt]
    """
    
    X = W[0]
    Y = W[1]
    Z = W[2]
    
    dXdt = -sigma * (X - Y)
    dYdt = r*X - Y - X*Z
    dZdt = -b*Z + X*Y
    
    return [dXdt, dYdt, dZdt]