import numpy as np, matplotlib.pyplot as plt, matplotlib.colors as colors

def solve_poisson(N=50, f_val=1.0):
    h, u, max_iter, tol = 1.0/(N-1), np.zeros((N,N)), 1000, 1e-6
    for it in range(max_iter):
        u_old = u.copy()
        for i in range(1,N-1):
            for j in range(1,N-1):
                u[i,j] = 0.25*(u_old[i+1,j]+u_old[i-1,j]+u_old[i,j+1]+u_old[i,j-1]+h**2*f_val)
        if np.max(np.abs(u-u_old)) < tol: break
    return u, h, it+1

def calculate_vector_field(u, h):
    gx, gy = np.gradient(u, h)
    gm = np.sqrt(gx**2+gy**2)
    sf = 0.8/np.max(gm) if np.max(gm)>0 else 1.0
    return gx, gy, gx*sf, gy*sf, gm

def calculate_border_velocity(u, h):
    gx, gy = np.gradient(u, h)
    hn = np.zeros_like(u)
    hn[0,:], hn[-1,:] = -gy[0,:], gy[-1,:]     # Bordas inferior/superior
    hn[:,0], hn[:,-1] = -gx[:,0], gx[:,-1]     # Bordas esquerda/direita
    return hn

def create_geometry_evolution():
    
    theta = np.linspace(0, 2*np.pi, 100)
    # Geometria inicial (quadrado suavizado)
    initial = 0.4 * (np.abs(np.cos(theta)) + np.abs(np.sin(theta))) / (np.abs(np.cos(theta)) + np.abs(np.sin(theta)) + 1e-10)
    # Geometria otimizada (forma com recuo nos pontos medios)
    optimized = 0.3 * (1 + 0.15 * np.cos(4*theta))  # Padrao de cruz com 4 pétalas
    return theta, initial, optimized

# Execução principal
N = 50
u_sol, h, iters = solve_poisson(N)
x = y = np.linspace(0,1,N)
X, Y = np.meshgrid(x,y)
gx, gy, gxn, gyn, gm = calculate_vector_field(u_sol, h)
hn_border = calculate_border_velocity(u_sol, h)

# Visualização
fig, axes = plt.subplots(1,3,figsize=(15,5))

#  Solução u(x,y)
im1 = axes[0].contourf(X,Y,u_sol,40,cmap='viridis')
axes[0].set_title('Solução $u(x,y)$'); axes[0].set_xlabel('$x$'); axes[0].set_ylabel('$y$')
plt.colorbar(im1, ax=axes[0])

#  Campo vetorial
skip = 2
quiver = axes[1].quiver(X[::skip,::skip],Y[::skip,::skip],gxn[::skip,::skip],gyn[::skip,::skip],
                       gm[::skip,::skip],cmap='plasma',scale=25,width=0.005,angles='xy')
axes[1].set_title('Campo vetorial $\\nabla u$'); axes[1].set_xlabel('$x$'); axes[1].set_ylabel('$y$')
plt.colorbar(quiver, ax=axes[1])

#  Evolução da geometria modificada
theta, init_geom, opt_geom = create_geometry_evolution()
ax_polar = fig.add_subplot(133, polar=True)
ax_polar.plot(theta, init_geom, 'k--', linewidth=2, label='Contorno inicial ($\\Omega_0$)')
ax_polar.plot(theta, opt_geom, 'r-', linewidth=2, label='Contorno otimizado ($\\Omega^*$)')
ax_polar.fill(theta, opt_geom, alpha=0.2, color='red')
ax_polar.set_title('Evolução da Geometria')
ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax_polar.grid(True)

plt.tight_layout(); plt.savefig('resultados_completos.png',dpi=300); plt.show()

# Análise de convergência
N_vals = [10,20,40,80]; center_vals = []; iters_list = []
for n in N_vals:
    u_temp, _, iters_temp = solve_poisson(n)
    center_vals.append(u_temp[n//2,n//2]); iters_list.append(iters_temp)

# Plot convergência
plt.figure(figsize=(10,4))
plt.subplot(121); plt.plot(N_vals, center_vals, 'o-'); plt.title('Valor no centro'); plt.grid(True)
plt.subplot(122); plt.plot(N_vals, iters_list, 's-'); plt.title('Iterações necessárias'); plt.grid(True)
plt.savefig('convergencia.png', dpi=300); plt.show()

print(f"=== RESULTADOS NUMÉRICOS ===")
print(f"u_max = {np.max(u_sol):.6f} (centro)")
print(f"u_borda = {u_sol[0,0]:.6f} (condição de contorno)")
print(f"Gradiente máximo = {np.max(gm):.6f}")
print(f"Iterações = {iters}")
print(f"Simetria: u(0.3,0.3)={u_sol[15,15]:.6f}, u(0.7,0.7)={u_sol[35,35]:.6f}")
