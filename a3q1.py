import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint,solve_ivp
# Problem 1(i)
def f1(y,t):
    return t*np.exp(3*t)-2*y
def exact1(t):
    return (1/5)*t*np.exp(3*t)-(1/25)*np.exp(3*t)+(1/25)*np.exp(-2*t)
t1=np.linspace(0,1,100)
y0_1=0
# Using odeint
sol_odeint_1=odeint(f1,y0_1,t1).flatten()
# Using solve_ivp
sol_solve_ivp_1=solve_ivp(lambda t,y: f1(y,t),[0,1],[y0_1],t_eval=t1).y[0]
# Exact solution
y_exact1=exact1(t1)
# Plotting Problem 1(i)
plt.figure(figsize=(10,5))
plt.plot(t1,y_exact1,'k--',label='Exact')
plt.plot(t1,sol_odeint_1,'b-',label='odeint')
plt.plot(t1,sol_solve_ivp_1,'r--',label='solve_ivp')
plt.title("Problem 1(i): $y'=te^{3t}-2y$")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
# Error plot
plt.figure()
plt.plot(t1,np.abs(y_exact1-sol_odeint_1),label='Error (odeint)')
plt.plot(t1,np.abs(y_exact1-sol_solve_ivp_1),label='Error (solve_ivp)')
plt.title("Error Comparison for Problem 1(i)")
plt.xlabel("t")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(True)
plt.show()
# Problem 1(ii)
def f2(y,t):
    return 1+(t-y)**2
def exact2(t):
    return t+1/(1-t)
t2=np.linspace(2,2.99,100)
y0_2=1
# Using odeint
sol_odeint_2=odeint(f2,y0_2,t2).flatten()
# Using solve_ivp
sol_solve_ivp_2=solve_ivp(lambda t,y: f2(y,t),[2,2.99],[y0_2],t_eval=t2).y[0]
# Exact solution
y_exact2=exact2(t2)
# Plotting Problem 1(ii)
plt.figure(figsize=(10, 5))
plt.plot(t2,y_exact2,'k--',label='Exact')
plt.plot(t2,sol_odeint_2,'b-',label='odeint')
plt.plot(t2,sol_solve_ivp_2,'r--',label='solve_ivp')
plt.title("Problem 1(ii): $y'=1+(t-y)^2$")
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
# Error plot
plt.figure()
plt.plot(t2,np.abs(y_exact2-sol_odeint_2),label='Error (odeint)')
plt.plot(t2,np.abs(y_exact2-sol_solve_ivp_2),label='Error (solve_ivp)')
plt.title("Error Comparison for Problem 1(ii)")
plt.xlabel("t")
plt.ylabel("Absolute Error")
plt.legend()
plt.grid(True)
plt.show()