import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Define the competition model
def competition_model(t,z):
    x,y=z
    dxdt=x*(2-0.4*x-0.3*y)
    dydt=y*(1-0.1*y-0.3*x)
    return [dxdt,dydt]
# Initial conditions for the 4 cases
initial_conditions=[[1.5,3.5],[1,1],[2,7],[4.5, 0.5]]
t_span=np.linspace(0,50,1000)
plt.subplots(figsize=(8,6))
plt.title('Population over Time (Competition Model)')
# Solve and plot each case
for i, z0 in enumerate(initial_conditions):
    sol=solve_ivp(competition_model,[0,50], z0, t_eval=t_span)
    t=sol.t
    x,y=sol.y
    plt.subplot(2,2,i+1)
    plt.plot(t,x,label='Predator',color='red')
    plt.plot(t,y,label='Prey',color='blue')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
plt.show()