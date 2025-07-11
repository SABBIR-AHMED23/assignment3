import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
def pendulum(t,z):
    theta,theta_dot=z
    dtheta_dt=theta_dot
    dtheta_dot_dt=-(32.17/2)*np.sin(theta)
    return [dtheta_dt,dtheta_dot_dt]
# Initial conditions
theta0=np.pi/6
theta_dot0=0
t_span=np.arange(0,2.1,0.1)
# Solve the system
sol=solve_ivp(pendulum, [0,2], [theta0,theta_dot0], t_eval=t_span,method='RK45')
t=sol.t
theta=sol.y[0]
# Display results
for i, j in zip(t,theta):
    print(f"t= {i:.1f}, theta= {j:.6f} rad")
# Plot theta vs time
plt.figure(figsize=(8,4))
plt.plot(t,theta,color='b',label=r'$\theta(t)$')
plt.title("Pendulum Motion: Angular Displacement vs Time")
plt.xlabel("Time (s)")
plt.ylabel('Theta (rad)')
plt.grid(True)
plt.legend()
plt.show()