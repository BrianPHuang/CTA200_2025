import numpy as np
import matplotlib.pyplot as plt

IMAX = 541
JMAX = 201

log_d_min = -12.0 # Minimum log density, log(rho), on the table
log_d_max = 15.0  # Maximum log density, log(rho), on the table

log_t_min = 3.0   # Minimum log time, log(t), on the table
log_t_max = 13.0  # Maximum log time, log(t), on the table

COLUMNS = [
    "f",       # F(rho, T)
    "fd",      # dF/d(rho)
    "ft",      # dF/dT
    "fdd",     # d^2 F / d(rho)^2
    "ftt",     # d^2 F / dT^2
    "fdt",     # d^2 F / d(rho) dT, equivalently, d^2 F / dT d(rho). Partial derivative commute
    "fddt",    # d^3 F / d(rho)^2 dT
    "fdtt",    # d^3 F / d(rho) dT^2
    "fddtt",   # d^4 F / d(rho)^2 dT^2
]

def load_helm_table(filename="helmholtz/helm_table.dat"):
    """
    Parameters
    ----------
    filename : str
        Path to helm_table.dat.
 
    Returns
    -------
    table : dict
        Keys 'f', 'fd', 'ft', 'fdd', 'ftt', 'fdt', 'fddt', 'fdtt', 'fddtt'
        -- each a 2D array of shape (IMAX, JMAX) indexed as [i_rho, j_T].
        Plus 'log_rho' (length IMAX) and 'log_T' (length JMAX) grid
        coordinates.
    """
    
    block_size = IMAX * JMAX
    raw = np.loadtxt(filename, max_rows=block_size)
    print(f"  Block 1 shape: {raw.shape}")
    
    if raw.shape[0] != block_size:
        raise ValueError(f"Got {raw.shape[0]} rows, expected IMAX*JMAX = {block_size}.")
    if raw.shape[1] != 9:
        raise ValueError(f"Got {raw.shape[1]} columns, expected 9.")
    
    table = {}
    for k, name in enumerate(COLUMNS):
        # File is row-major with density varying fastest, so reshape
        table[name] = raw[:, k].reshape((JMAX, IMAX)).T # to (JMAX, IMAX) then transpose -> (IMAX, JMAX).
     
    table["log_rho"] = np.linspace(log_d_min, log_d_max, IMAX)
    table["log_T"] = np.linspace(log_t_min, log_t_max, JMAX)
    
    print(f"  Loaded 9 arrays of shape ({IMAX}, {JMAX}).")
    print(f"  log_rho: [{log_d_min}, {log_d_max}]"
          f"  ({IMAX} points, "
          f"{(log_d_max - log_d_min)/(IMAX-1):.4f} per step)")
    print(f"  log_T  : [{log_t_min}, {log_t_max}]"
          f"  ({JMAX} points, "
          f"{(log_t_max - log_t_min)/(JMAX-1):.4f} per step)")
    
    return table
    
    
def sanity_check_plot(table, savefig=True):
    """Plot F, dF/d(rho), dF/dT as signed-log heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
 
    log_rho = table["log_rho"]
    log_T = table["log_T"]
 
    panels = [
        ("f",  r"$F$"),
        ("fd", r"$\partial F / \partial \rho$"),
        ("ft", r"$\partial F / \partial T \; = \; -S$"),
    ]
 
    for ax, (key, label) in zip(axes, panels):
        arr = table[key].T  # pcolormesh wants Z[y, x] = Z[T, rho]
        signed_log = np.sign(arr) * np.log10(np.abs(arr) + 1e-300)
        im = ax.pcolormesh(log_rho, log_T, signed_log,
                           shading="auto", cmap="RdBu_r")
        ax.set_xlabel(r"$\log_{10}(\rho\ /\ \mathrm{g\,cm^{-3}})$")
        ax.set_ylabel(r"$\log_{10}(T\ /\ \mathrm{K})$")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, label=r"$\mathrm{sgn}\cdot\log_{10}|\cdot|$")
 
    plt.tight_layout()
    if savefig:
        plt.savefig("helm_table_sanity.pdf")
        print("Saved helm_table_sanity.pdf")
    plt.show()
 
 
if __name__ == "__main__":
    table = load_helm_table("helm_table.dat")
 
    print("\nQuick stats on each quantity in Block 1:")
    for key in COLUMNS:
        a = table[key]
        print(f"  {key:>6s}:  min={a.min():+.3e}  max={a.max():+.3e}")
 
    sanity_check_plot(table)
    
    
    
    
    
    
    
    
    
    
    
    
