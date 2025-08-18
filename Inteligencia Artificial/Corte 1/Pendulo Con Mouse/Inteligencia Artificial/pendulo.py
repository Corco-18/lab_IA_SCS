import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation


g = 9.81   
l = 1.0      
m = 1.0     
b = 0.1      

# Ecuaciones del péndulo invertido
def deriv(y, t):
    theta, omega = y
    dydt = [omega, (g / l) * np.sin(theta) - b * omega]
    return dydt

# Condiciones iniciales: pequeño desplazamiento
y0 = [0.1, 0.0]
t = np.linspace(0, 10, 300)
sol = odeint(deriv, y0, t)

# Animación
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot([], [], 'o-', lw=2)

def animate(i):
    x = l * np.sin(sol[i, 0])
    y = -l * np.cos(sol[i, 0])
    line.set_data([0, x], [0, y])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=33, blit=True)
plt.show()
