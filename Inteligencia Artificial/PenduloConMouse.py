import numpy as np
import pygame

# -------------------------------
# Modelo del péndulo invertido con fricción
# -------------------------------
def pendulo_invertido_modelo(state, F, params, dt):
    x, dx, theta, dtheta = state
    M = params["M"]
    m = params["m"]
    L = params["L"]
    g = params["g"]
    b = params["b"]
    mu = params["mu"]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denominator = M + m - m * cos_theta**2

    ddx = (F - b * dx + m * sin_theta * (L * dtheta**2 + g * cos_theta - mu * dtheta * cos_theta / (m * L))) / denominator
    ddtheta = (-F * cos_theta + b * dx * cos_theta - (M + m) * g * sin_theta - mu * dtheta * (M + m) / (m * L)) / (L * denominator)

    # Integración numérica (Euler)
    x += dx * dt
    dx += ddx * dt
    theta += dtheta * dt
    dtheta += ddtheta * dt

    return np.array([x, dx, theta, dtheta])

# -------------------------------
# Parámetros físicos con fricción
# -------------------------------
params = {
    "M": 0.8,
    "m": 0.4,
    "L": 0.4,
    "g": 9.81,
    "b": 0.05,
    "mu": 0.03
}

# -------------------------------
# Inicialización de Pygame
# -------------------------------
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido Controlado con el Mouse")
clock = pygame.time.Clock()

# Estado inicial
state = np.array([0.0, 0.0, np.pi, 0.0])
dt = 0.02
k_mouse = 3.5  # constante de proporcionalidad de fuerza con el mouse

font = pygame.font.SysFont("Arial", 20)
running = True
escala_pixeles = 200  # 1 m = 200 px

# -------------------------------
# Bucle principal
# -------------------------------
while running:
    screen.fill((255, 255, 255))

    # ---------------------------
    # Control por posición del mouse
    # ---------------------------
    mouse_x, _ = pygame.mouse.get_pos()
    x_pix = int(WIDTH // 2 + state[0] * escala_pixeles)
    distancia_pix = mouse_x - x_pix
    F = k_mouse * distancia_pix / escala_pixeles  # convierte a Newtons aprox

    # ---------------------------
    # Simulación
    # ---------------------------
    state = pendulo_invertido_modelo(state, F, params, dt)

    # ---------------------------
    # Dibujos
    # ---------------------------
    x_pix = int(WIDTH // 2 + state[0] * escala_pixeles)
    cart_y = HEIGHT // 2
    pend_x = int(x_pix + params["L"] * escala_pixeles * np.sin(state[2]))
    pend_y = int(cart_y + params["L"] * escala_pixeles * np.cos(state[2]))

    # Carro
    pygame.draw.rect(screen, (0, 0, 0), (x_pix - 25, cart_y - 10, 50, 20))
    # Péndulo
    pygame.draw.line(screen, (255, 0, 0), (x_pix, cart_y), (pend_x, pend_y), 5)
    # Eje
    pygame.draw.circle(screen, (0, 0, 255), (x_pix, cart_y), 5)
    # Masa del péndulo (radio proporcional a m^(1/3))
    masa_radio = int(10 * (params["m"])**(1/3))
    pygame.draw.circle(screen, (0, 200, 0), (pend_x, pend_y), masa_radio)

    # ---------------------------
    # Texto con variables físicas
    # ---------------------------
    labels = [
        f"x = {state[0]:.2f} m",
        f"dx = {state[1]:.2f} m/s",
        f"θ = {np.degrees(state[2]) % 360:.2f}°",
        f"dθ = {np.degrees(state[3]):.2f} °/s",
        f"F = {F:.2f} N"
    ]
    for i, label in enumerate(labels):
        text = font.render(label, True, (0, 0, 0))
        screen.blit(text, (10, 10 + i * 25))

    pygame.display.flip()
    clock.tick(50)

    # ---------------------------
    # Eventos
    # ---------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
