import numpy as np
import pygame
from scipy.integrate import odeint

# ==== Parámetros físicos ====
M = 0.5      # masa del carrito (kg)
m = 0.2      # masa del péndulo (kg)
b = 0.1      # fricción del carrito
l = 0.3      # longitud al centro de masa del péndulo (m)
I = 0.006    # inercia del péndulo (kg*m^2)
g = 9.81     # gravedad (m/s²)

# ==== Parámetros PID ====
Kp_theta = 150.0
Ki_theta = 0.0
Kd_theta = 25.0

Kp_x = 1.0
Ki_x = 0.0
Kd_x = 1.0

# ==== Estado inicial ====
x = 0.0
x_dot = 0.0
theta = np.deg2rad(10)  # inclinación inicial (rad)
theta_dot = 0.0

state = np.array([x, x_dot, theta, theta_dot])

# ==== PID ====
error_sum_theta = 0.0
prev_error_theta = 0.0

error_sum_x = 0.0
prev_error_x = 0.0

# ==== Tiempo de simulación ====
dt = 0.02
t = 0.0

# ==== Inicializar Pygame ====
pygame.init()
WIDTH, HEIGHT = 1000, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido con PID")
clock = pygame.time.Clock()

# ==== Escalado para dibujo ====
scale = 200  # pixeles por metro
cart_y = HEIGHT // 2

# ==== Modelo dinámico ====
def deriv(y, t, F):
    x, x_dot, theta, theta_dot = y

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = I*(M+m) + M*m*l**2 + m**2*l**2*(1 - cos_theta**2)

    theta_ddot = (
        (m * g * l * sin_theta * (M + m) - cos_theta * (F + m * l * theta_dot**2 * sin_theta - b * x_dot)) /
        (l * denominator)
    )

    x_ddot = (
        F + m * l * theta_dot**2 * sin_theta - m * l * cos_theta * theta_ddot - b * x_dot
    ) / (M + m)

    return [x_dot, x_ddot, theta_dot, theta_ddot]

# ==== Bucle principal ====
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ==== PID para ángulo ====
    error_theta = 0.0 - state[2]  # queremos que theta sea 0
    error_sum_theta += error_theta * dt
    d_error_theta = (error_theta - prev_error_theta) / dt
    prev_error_theta = error_theta

    u_theta = (Kp_theta * error_theta) + (Ki_theta * error_sum_theta) + (Kd_theta * d_error_theta)

    # ==== PID para posición ====
    error_x = 0.0 - state[0]  # queremos que x sea 0
    error_sum_x += error_x * dt
    d_error_x = (error_x - prev_error_x) / dt
    prev_error_x = error_x

    u_x = (Kp_x * error_x) + (Ki_x * error_sum_x) + (Kd_x * d_error_x)

    # ==== Combinación de control ====
    F = u_theta + u_x

    # ==== Integración numérica ====
    state = odeint(deriv, state, [0, dt], args=(F,))[1]

    # ==== Dibujar ====
    screen.fill((255, 255, 255))

    cart_x = WIDTH // 2 + float(state[0] * scale)  # aseguramos que sea float

    # Carrito
    pygame.draw.rect(screen, (0, 0, 0), (cart_x - 40, cart_y - 20, 80, 40), 2)

    # Péndulo
    pend_x = cart_x + float(l * scale * np.sin(state[2]))
    pend_y = cart_y - float(l * scale * np.cos(state[2]))
    pygame.draw.line(screen, (255, 0, 0), (cart_x, cart_y), (pend_x, pend_y), 4)
    pygame.draw.circle(screen, (0, 0, 255), (int(pend_x), int(pend_y)), 10)

    pygame.display.flip()
    clock.tick(50)

pygame.quit()
