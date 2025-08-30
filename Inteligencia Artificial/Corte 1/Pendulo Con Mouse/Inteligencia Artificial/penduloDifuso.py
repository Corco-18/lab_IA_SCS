import pygame
import numpy as np

# =========================
# Parámetros físicos
# =========================
g = 9.81
L = 1.0   # Longitud del péndulo (m)
m = 1.0   # Masa del péndulo (kg)
M = 5.0   # Masa del carro (kg)
dt = 0.02

# =========================
# Conjuntos difusos trapezoidales
# =========================
def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal"""
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a+1e-6), 1), (d-x)/(d-c+1e-6)))

# Universo de discurso
angulo_universe = np.linspace(-0.5, 0.5, 100)   # rad
vel_universe = np.linspace(-2, 2, 100)          # rad/s
force_universe = np.linspace(-50, 50, 200)      # Fuerza N

# Definición de conjuntos difusos de entrada
AP  = [0.05, 0.15, 0.25, 0.35]
AMP = [0.25, 0.35, 0.5, 0.5]
AN  = [-0.35, -0.25, -0.15, -0.05]
AMN = [-0.5, -0.5, -0.35, -0.25]

VAN = [-2, -2, -0.5, 0]
VAP = [0, 0.5, 2, 2]

# Conjuntos difusos de salida
XN  = [-50, -50, -20, -5]
X0  = [-5, -2, 2, 5]
XP  = [5, 20, 50, 50]
XMP = [20, 35, 50, 50]
XMN = [-50, -50, -35, -20]

def fuzzify_angle(x):
    return {
        "ap": trapmf(x, *AP),
        "amp": trapmf(x, *AMP),
        "an": trapmf(x, *AN),
        "amn": trapmf(x, *AMN)
    }

def fuzzify_vel(x):
    return {
        "van": trapmf(x, *VAN),
        "vap": trapmf(x, *VAP)
    }

# Reglas difusas
rules = [
    ("ap", "vap", "xp"),
    ("ap", "van", "x0"),
    ("an", "vap", "x0"),
    ("an", "van", "xn"),
    ("amp", "vap", "xmp"),
    ("amp", "van", "xp"),
    ("amn", "van", "xn"),
    ("amn", "vap", "xmn"),
]

def apply_rules(angle_mf, vel_mf):
    outputs = []
    for rule in rules:
        ang_set, vel_set, out_set = rule
        w = min(angle_mf[ang_set], vel_mf[vel_set])
        if w > 0:
            if out_set == "xp":
                mf = np.minimum(w, trapmf(force_universe, *XP))
            elif out_set == "xn":
                mf = np.minimum(w, trapmf(force_universe, *XN))
            elif out_set == "x0":
                mf = np.minimum(w, trapmf(force_universe, *X0))
            elif out_set == "xmp":
                mf = np.minimum(w, trapmf(force_universe, *XMP))
            elif out_set == "xmn":
                mf = np.minimum(w, trapmf(force_universe, *XMN))
            outputs.append(mf)
    if outputs:
        return np.fmax.reduce(outputs)
    else:
        return np.zeros_like(force_universe)

def defuzzify(mf):
    if np.sum(mf) == 0:
        return 0
    return np.sum(force_universe * mf) / np.sum(mf)

# =========================
# Dinámica del sistema
# =========================
def dynamics(state, force):
    x, x_dot, theta, theta_dot = state
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * sin_theta**2

    x_ddot = (force + m*sin_theta*(L*theta_dot**2 + g*cos_theta)) / denom
    theta_ddot = (-force*cos_theta - m*L*theta_dot**2*cos_theta*sin_theta - (M+m)*g*sin_theta) / (L*denom)

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

# =========================
# Simulación Pygame
# =========================
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Control difuso - Péndulo invertido")
clock = pygame.time.Clock()

cart_x = 400
cart_y = 500

state = np.array([0.0, 0.0, 0.2, 0.0])  # x, x_dot, theta(rad), theta_dot
running = True
simulate = False

font = pygame.font.SysFont(None, 36)
button = pygame.Rect(650, 50, 100, 50)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button.collidepoint(event.pos):
                simulate = not simulate

    screen.fill((255, 255, 255))

    if simulate:
        angle_mf = fuzzify_angle(state[2])
        vel_mf = fuzzify_vel(state[3])
        output_mf = apply_rules(angle_mf, vel_mf)
        force = defuzzify(output_mf)

        k1 = dynamics(state, force)
        k2 = dynamics(state + 0.5*dt*k1, force)
        k3 = dynamics(state + 0.5*dt*k2, force)
        k4 = dynamics(state + dt*k3, force)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Dibujar carro
    cart_x = 400 + int(state[0]*100)
    pygame.draw.rect(screen, (0,0,0), (cart_x-40, cart_y-20, 80, 40), 2)

    # Dibujar péndulo
    pend_x = cart_x + int(L*100*np.sin(state[2]))
    pend_y = cart_y - int(L*100*np.cos(state[2]))
    pygame.draw.line(screen, (0,0,255), (cart_x, cart_y), (pend_x, pend_y), 4)
    pygame.draw.circle(screen, (255,0,0), (pend_x, pend_y), 10)

    # Botón RUN
    pygame.draw.rect(screen, (0,200,0), button)
    text = font.render("RUN" if not simulate else "STOP", True, (255,255,255))
    screen.blit(text, (button.x+10, button.y+10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
