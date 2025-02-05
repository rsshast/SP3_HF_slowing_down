import numpy as np
import matplotlib.pyplot as plt
from setup_sp3 import * 

# Integration bounds for scattering range.
# Take the MINIMUM of the bounds between u' (incident) and the minumum of g_min,u+ln(1/alpha)

grid = get_data(XS38,gridpoints,NU)
gridwidth = grid[1,0] - grid[0,0] #CONSTANT gridspacing

# Number of shading regions desired
shading_colors = ['lightgreen', 'yellow', 'blue']
m = len(shading_colors)

# Functions. u incident, u' outgoing
alpha = ((AU - 1) / (AU + 1)) ** 2
u_vals = np.linspace(0, m * gridwidth,200)
y_u    = u_vals  
y_u_ln = u_vals + np.log(1 / alpha)  # y = x + ln(1/alpha) (red line)

# Create the plot
plt.figure(figsize=(7, 7))
plt.plot(u_vals, y_u, 'black', label="Min: u' = u")
plt.plot(u_vals, y_u_ln, 'red', label=r"Max: u' = u + $\ln(1/\alpha)$")

for n in range(m + 1):  # for each shaded region
    grid_y = n * gridwidth
    grid_x = n * gridwidth

    # Plot dashed grid lines
    plt.axhline(grid_y, color='gray', linestyle='dashed')
    plt.axvline(grid_x, color='gray', linestyle='dashed')

    # Clip shading strictly between y = u and y = u + ln(1/alpha), respecting grid lines
    if n < m:
        mask = (u_vals >= grid_x)  # No shading to the left of the vertical grid line
        y_lower = np.maximum(y_u, grid_y)  # Ensure shading starts at max(grid line, y_u)
        y_upper = np.minimum(y_u_ln, (n + 1) * gridwidth)  # Ensure shading stops at min(y_u_ln, next grid line)
        valid_shading = (y_upper > y_lower) & mask # Apply mask to ensure shading only occurs in the valid region
        plt.fill_between(u_vals, y_lower, y_upper, where=valid_shading, color=shading_colors[n], alpha=0.5) # Fill

plt.xlabel("u")
plt.ylabel("u'")
plt.xlim(0,m*gridwidth)
plt.ylim(0,m*gridwidth)
plt.title("Integration Bounds")
plt.legend()
plt.savefig("results/charts/integration_bounds.png")
