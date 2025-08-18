import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros físicos
M = 0.5     # masa del carrito (kg)
m = 0.2     # masa del péndulo (kg)
b = 0.1     # fricción del carrito
l = 1     # longitud al centro de masa del péndulo (m)
I = 0.006   # inercia del péndulo (kg*m^2)
g = 9.81    # gravedad (m/s²)

# Ecuaciones del sistema (no controlado)
def deriv(y, t):
    x, x_dot, theta, theta_dot = y

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = I*(M+m) + M*m*l**2 + m**2*l**2*(1 - cos_theta**2)

    theta_ddot = (
        (m * g * l * sin_theta * (M + m) - cos_theta * (m * l * theta_dot**2 * sin_theta - b * x_dot)) /
        (l * denominator)
    )

    x_ddot = (
        m * l * theta_dot**2 * sin_theta - m * l * cos_theta * theta_ddot - b * x_dot
    ) / (M + m)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# Condiciones iniciales: péndulo levemente inclinado, carrito en reposo
y0 = [0.0, 0.0, np.pi - 0.1, 0.0]  # [x, x_dot, theta, theta_dot]

# Tiempo de simulación
t = np.linspace(0, 10, 400)

# Resolver el sistema
sol = odeint(deriv, y0, t)
x = sol[:, 0]
theta = sol[:, 2]

# Animación
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-0.5, 1.5)
ax.set_aspect('equal')
cart_width = 0.4
cart_height = 0.2
line, = ax.plot([], [], 'o-', lw=2)
cart = plt.Rectangle((0, 0), cart_width, cart_height, fc='blue')
ax.add_patch(cart)

def animate(i):
    cart.set_xy((x[i] - cart_width / 2, 0))
    pendulum_x = x[i] + l * np.sin(theta[i])
    pendulum_y = cart_height + l * np.cos(theta[i])
    line.set_data([x[i], pendulum_x], [cart_height, pendulum_y])
    return cart, line

ani = animation.FuncAnimation(fig, animate, frames=len(t), interval=25, blit=True)
plt.show()
