import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,solve_bvp
# Define constants
a,b=0,1
ya,yb=1,np.exp(-10)
def ode_system(x,y):
    y1,y2=y
    dy1=y2
    dy2=100*y1
    return [dy1,dy2]
# Linear shooting method
def linear_shooting(h):
    N=int((b-a)/h)
    x_vals=np.linspace(a,b,N+1)
    # First IVP: y1'' = 100*y1, y1(0) = 1, y1'(0) = 0
    sol1=solve_ivp(ode_system,[a,b], [1,0], t_eval=x_vals)
    # Second IVP: y2'' = 100*y2, y2(0) = 0, y2'(0) = 1
    sol2=solve_ivp(ode_system,[a,b], [0,1], t_eval=x_vals)
    # Interpolate solution using shooting formula
    beta=yb
    y1b=sol1.y[0,-1]
    y2b=sol2.y[0,-1]
    c=(beta-y1b)/y2b
    y_approx=sol1.y[0]+c*sol2.y[0]
    plt.plot(x_vals,y_approx,label=f"Shooting h={h}")
    return x_vals, y_approx
# Solve using solve_bvp
def bvp_solution():
    def fun(x,y):
        return np.vstack((y[1], 100*y[0]))
    def bc(ya,yb):
        return np.array([ya[0]-1, yb[0]-np.exp(-10)])
    x=np.linspace(a,b,100)
    y_init=np.zeros((2, x.size))
    sol=solve_bvp(fun,bc, x, y_init)
    plt.plot(sol.x, sol.y[0],'k--',label='solve_bvp')
    return sol.x, sol.y[0]
# Plotting solutions
plt.figure(figsize=(10,6))
linear_shooting(0.1)
linear_shooting(0.05)
bvp_solution()
plt.plot(np.linspace(0,1,100),np.exp(-10*np.linspace(0,1,100)),'g:',label='Exact $e^{-10x}$')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('BVP Solution Comparison')
plt.grid(True)
plt.show()