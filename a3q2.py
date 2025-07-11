import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# Define the system
def predator_prey(t,z):
    x,y=z
    dxdt=-0.1*x+0.02*x*y
    dydt=0.2*y-0.025*x*y
    return [dxdt,dydt]
# Initial conditions
x0=6
y0=6
t_span=np.linspace(0,50,1000)
# Solve the system
sol=solve_ivp(predator_prey,[0,50],[x0,y0],t_eval=t_span)
t=sol.t
x=sol.y[0]
y=sol.y[1]
# Plot x(t) and y(t)
plt.figure(figsize=(10,5))
plt.plot(t,x,label='Predators (x)',color='red')
plt.plot(t,y,label='Prey (y)',color='green')
plt.title("Predator-Prey Model")
plt.xlabel("Time (t)")
plt.ylabel("Population (thousands)")
plt.legend()
plt.grid(True)
plt.show()