from Sp3 import Sp3
from setup_sp3 import *
from scipy.special import legendre

#get data
print("Reading Data...")
t_data = time.time()
chi = get_data(chi,gridpoints,NU)
H = get_data(H,gridpoints,NH)
H *= NH
XS38 = get_data(XS38,gridpoints,NU)
sigma_f = get_fission_data(XS38[:,0],sigma_f)

sigma_s = np.flip(XS38[:,2])
sigma_t = np.flip(XS38[:,1])
chi = np.flip(chi[:,1])
print(np.sum(chi))
assert 0 == 1
groups = XS38[:,0]
E = np.exp(np.flip(groups))

def compute_mu_matrix(lethargy, A):
    """
    Compute scattering cosine matrix mu(u, u') for all g, g' combinations.
    """
    u = lethargy.reshape(-1, 1)     # (G,1)
    up = lethargy.reshape(1, -1)    # (1,G)
    mu = ((A + 1) / 2) * np.exp((up - u) / 2) - ((A - 1) / 2) * np.exp((u - up) / 2)
    mu = np.clip(mu, -1, 1)
    return mu

def compute_sigma_s_moments(sigma_s, mu_matrix, lethargy, A):
    """
    Compute sigma_s_l matrices for l = 0 to 3 using Legendre projections,
    and apply a kinematic cutoff using alpha.
    """
    G = len(lethargy)
    sigma_s = sigma_s.reshape(1, -1)  # shape (1, G) for broadcasting
    sigma_s_l = []

    # Kinematic cutoff: u > u' + ln(1/alpha)
    alpha = ((A - 1) / (A + 1))**2
    delta_umax = np.log(1 / alpha)

    u = lethargy.reshape(-1, 1)  # (G, 1)
    up = lethargy.reshape(1, -1) # (1, G)
    allowed = u <= (up + delta_umax)  # (G, G) mask

    for l in range(4):
        P_l = legendre(l)(mu_matrix)
        raw_sigma = P_l * sigma_s
        raw_sigma[~allowed] = 0.0
        sigma_s_l.append(raw_sigma)

    return sigma_s_l  # list of (G,G) arrays

def compute_Ln_matrix(n, sigma_t, sigma_s_l):
    """
    Compute L_n matrix = (2n+1)(Σ_t - Σ_s_l) for each moment n.
    """
    G = len(sigma_t)
    Sigma_t_mat = np.diag(sigma_t)
    L_n = (2 * n + 1) * (Sigma_t_mat - sigma_s_l)
    return L_n

def compute_all_Ln(lethargy, sigma_t, sigma_s, A):
    """
    Compute L0 to L3 matrices for SP3 equations with cutoff applied.
    """
    mu_matrix = compute_mu_matrix(lethargy, A)
    sigma_s_l = compute_sigma_s_moments(sigma_s, mu_matrix, lethargy, A)

    L_matrices = {}
    for n in range(4):
        Ln = compute_Ln_matrix(n, sigma_t, sigma_s_l[n])
        L_matrices[f'L{n}'] = Ln

    return L_matrices, sigma_s_l

print("L mats")
L_matrices, sigma_s_l = compute_all_Ln(groups, sigma_t, sigma_s, AU)

# Access each matrix
L0 = L_matrices['L0']
L1 = L_matrices['L1']
L2 = L_matrices['L2']
L3 = L_matrices['L3']

B4 = B2**2
I = np.eye(len(groups))

# Ensure uniform lethargy grid spacing Δu
delta_u = np.gradient(groups)  # groups is lethargy grid (ascending)

# Normalize chi with respect to Δu
chi = chi / np.sum(chi * delta_u)

# Identity
I = np.eye(len(groups))
B4 = B2 ** 2

print("lin alg")
# Form operators (with implicit Δu if needed)
LHS = (
    9 * B4 * I
    + B2 * (L3 @ L2 + (9 * L1 + 4 * L3) @ L0)
    + (L3 @ L2 @ L1 @ L0)
)

RHS = (L3 @ L2 @ L1 + B2 * (9 * L1 + 4 * L3)) @ chi

W = np.diag(delta_u)
LHS_weighted = W @ LHS @ W
RHS_weighted = W @ RHS
phi0 = np.linalg.solve(LHS_weighted, RHS_weighted)
#phi0 = np.clip(phi0, 0, None)
#phi0 /= np.trapz(phi0, x=E)

plt.figure()
plt.plot(E,phi0,label=r'$\phi_0$')
plt.title(r'$\phi_0(E)$')
plt.xlabel('E')
plt.ylabel(r'$\phi_0$')
plt.xscale('log')
plt.grid(True, which='both')
plt.legend()
plt.savefig(f'phi0.png')

# Visualize sparsity in sigma_s0 (optional)
#plt.imshow(sigma_s_l[0], aspect='auto', origin='upper', cmap='viridis')
#plt.title(r"$\Sigma_{s0}(g' \to g)$ with lethargy cutoff")
#plt.xlabel("Incoming group $g'$ (u')")
#plt.ylabel("Outgoing group $g$ (u)")
#plt.colorbar(label="Scattering XS")
#plt.tight_layout()
#plt.savefig("sigma_s0.png")

print(f'Computation time = {np.round(et - st,5)}s')
