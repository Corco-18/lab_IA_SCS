import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Parámetros físicos
M = 1.0     # masa del carrito
m = 0.1     # masa del péndulo
l = 1.0     # longitud del péndulo
g = 9.81    # gravedad

# Ecuaciones del sistema
def dynamics(t, y):
    theta, omega, x, v = y
    u = 0  # fuerza de control nula por ahora

    # Ecuaciones dadas
    theta_ddot = ((M + m) * g * theta - u) / (M * l)
    x_ddot = (u - m * g * theta) / M

    return [omega, theta_ddot, v, x_ddot]

# Condiciones iniciales: [theta, theta_dot, x, x_dot]
y0 = [np.radians(10), 0, 0, 0]

# Tiempo de simulación
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 500)

# Resolver sistema
sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval)

theta = sol.y[0]
x = sol.y[2]

# Animación
fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
cart, = ax.plot([], [], 'k-', lw=5)
pendulum, = ax.plot([], [], 'o-', lw=2)

def init():
    cart.set_data([], [])
    pendulum.set_data([], [])
    return cart, pendulum

def update(i):
    cart_width = 0.4
    cart_x = [x[i] - cart_width, x[i] + cart_width]
    cart_y = [0, 0]
    cart.set_data(cart_x, cart_y)

    pend_x = [x[i], x[i] + l * np.sin(theta[i])]
    pend_y = [0, -l * np.cos(theta[i])]
    pendulum.set_data(pend_x, pend_y)

    return cart, pendulum

ani = FuncAnimation(fig, update, frames=len(t_eval),
                    init_func=init, blit=True, interval=20)

plt.title("Péndulo Invertido (sin control)")
plt.xlabel("Posición del carrito")
plt.ylabel("Altura")
plt.grid()
plt.show()
