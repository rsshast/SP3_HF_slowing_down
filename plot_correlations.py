import numpy as np
import matplotlib.pyplot as plt

groups = np.array([5000, 10000,15000,25000,30000,50000])
B2 = np.array([0.332, 1.18,1.67,2.11,2.21,2.28])
L2 = np.array([0.0480649, 0.0445713, 0.0406698, 0.0308709, 0.0256858, 0.0144072])

# Exponential fit for L2 
log_y = np.log(L2)
b_exp, log_a_exp = np.polyfit(groups, log_y, 1)
a_exp = np.exp(log_a_exp)
L2_fit = a_exp * np.exp(b_exp * groups)
# R2 for exponential
ss_res = np.sum((L2 - L2_fit)**2)
ss_tot = np.sum((L2 - np.mean(L2))**2)
r2_exp = 1 - ss_res/ss_tot


# Logarithmic fit for B2
log_x = np.log(groups)
a_log, b_log = np.polyfit(log_x, B2, 1)
B2_fit = a_log*np.log(groups) + b_log
# R2 for log
ss_res2 = np.sum((B2 - B2_fit)**2)
ss_tot2 = np.sum((B2 - np.mean(B2))**2)
r2_log = 1 - ss_res2/ss_tot2

# trendlines
x_smooth = np.linspace(groups.min(), groups.max(), 200)
L2_smooth = a_exp * np.exp(b_exp * x_smooth)
B2_smooth = a_log*np.log(x_smooth) + b_log

# plot
fig, ax1 = plt.subplots()

# L2
ax1.scatter(groups, L2, label=r'$\mathcal{L}_2$ Norm', color = 'blue')
ax1.plot(x_smooth, L2_smooth, color = 'blue', linestyle='--')
ax1.set_xlabel("Number of Energy Bins")
ax1.set_ylabel(r'$\mathcal{L}_2$ Norm', color='blue')

# B2 
ax2 = ax1.twinx()
ax2.scatter(groups, B2,color='orange',label=r'$B^2$')
ax2.plot(x_smooth, B2_smooth, color='orange', linestyle='--')
ax2.set_ylabel("B2 Error (%)", color='orange')


# Equation text 
eq1 = rf"$L_2 = {a_exp:.3e}e^{{{b_exp:.3e}x}}$" + "\n" + rf"$R^2 = {r2_exp:.4f}$"
eq2 = rf"$B_2 = {a_log:.3f}\ln(x) + {b_log:.3f}$" + "\n" + rf"$R^2 = {r2_log:.4f}$"
ax1.text(0.20,0.975,eq1, transform=ax1.transAxes, va='top')
ax2.text(0.60,0.50,eq2, transform=ax2.transAxes)

# touch up
plt.grid(which='Both')
plt.tight_layout()
plt.savefig("results/charts/b2_eqns.png")
