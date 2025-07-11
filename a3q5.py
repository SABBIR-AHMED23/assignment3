import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Define the system of first-order ODEs
def system(t,y):
    y0,y1,y2,y3,y4,y5=y
    dy0=y1
    dy1=y2
    dy2=-2*y4**2+y3  
    dy3=y4
    dy4=y5
    dy5=-y2**3+y4+y0+np.sin(t) 
    return [dy0,dy1,dy2,dy3,dy4,dy5]
# Initial conditions
y0_init=[1,0,0,1,0,0]
t_span=np.linspace(0,10,1000)
# Solve the system
sol=solve_ivp(system,[0,10], y0_init,t_eval=t_span)
# Plot the solutions
plt.plot(sol.t, sol.y[0],label='$x_1(t)$')
plt.plot(sol.t, sol.y[3],label='$x_2(t)$')
plt.xlabel('t')
plt.ylabel('Solution')
plt.title('Solution of the System of ODEs')
plt.legend()
plt.grid(True)
plt.show()