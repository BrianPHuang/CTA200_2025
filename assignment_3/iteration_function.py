import numpy as np

def mandelbrot_iterations(x_min=-2, x_max=2,
                          y_min=-2, y_max=2,
                          n_pts=1000, max_iter=100):
    """
    Compute iteration results

    Parameters

    ----------

    x_min, x_max : float, Range of the real part.

    y_min, y_max : float, Range of the imaginary part.

    n_pts : int, Number of grid points in each direction.

    max_iter : int, Maximum number of iterations.

    Returns

    -------

    x_vals : numpy array, Real-axis values.

    y_vals : numpy array, Imaginary-axis values.

    bound : 2D numpy array, True if the point stays bounded up to max_iter.

    diverg_iter : 2D numpy array, Iteration number when each point diverges. Points that do not diverge are set to max_iter.
    """
    
    x_vals = np.linspace(x_min, x_max, n_pts)
    y_vals = np.linspace(y_min, y_max, n_pts)

    diverge_iter = np.zeros((n_pts, n_pts), dtype = int)
    bound = np.ones((n_pts, n_pts), dtype = int)
    
    for i in range(n_pts):
        for j in range(n_pts):
            c = x_vals[j] + 1j * y_vals[i] # Ddefine 1j ass the unit in imaginary axis
            z_n = 0
            for iter in range(max_iter):
                z_n = z_n**2 + c

                if abs(z_n) > 2: # Exceeding the limit --> Diverge
                    bound[i, j] = False
                    diverge_iter[i, j] = iter
                    break

            if bound[i, j]:
                diverge_iter[i, j] = max_iter

    return x_vals, y_vals, bound, diverge_iter
