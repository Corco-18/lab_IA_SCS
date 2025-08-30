
import numpy as np
import pygame
import math

# Parámetros 
M = 0.5      # masa carro (kg)
m = 0.2      # masa péndulo (kg)
b = 0.1      # fricción carro (N·s/m)
l = 0.5    # distancia CG (m)
I = 0.006    # inercia péndulo
b_p = 0.01   # fricción pivote
g = 9.81
dt = 0.01

#Constantes PID
Kp_theta = 150.0
Ki_theta = 0.0
Kd_theta = 25.0

# Referencia Cascada
Kp_pos = 2.0
Kd_pos = 1.2

# Límites
force_limit = 25.0   # Limite fuerza del motor
theta_ref_limit = math.radians(18)  # limite del angulo
x_limit = 3  # m, maximo que el carro puede moverse

def initial_state():
    # estado: [x, x_dot, Angulo Inicial, theta_dot]  
    return np.array([0.0, 0.0, math.radians(8.0), 0.0], dtype=float)

state = initial_state()
integral_theta = 0.0


def f_nonlinear(y, F):
    x, x_dot, th, th_dot = y
    st = math.sin(th)
    ct = math.cos(th)
    Den = (I + m * l**2) * (M + m) - (m * l * ct)**2
    if abs(Den) < 1e-9:
        Den = 1e-9

    Fx = F - b * x_dot - m * l * th_dot**2 * st
    T = m * g * l * st + b_p * th_dot

    th_dd = ((m * l * ct) * Fx - (M + m) * T) / Den
    x_dd  = ((I + m * l**2) * Fx + (m * l * ct) * T) / Den
    return np.array([x_dot, x_dd, th_dot, th_dd], dtype=float)

# RK4 integrator step
def rk4_step(y, F, dt):
    k1 = f_nonlinear(y, F)
    k2 = f_nonlinear(y + 0.5*dt*k1, F)
    k3 = f_nonlinear(y + 0.5*dt*k2, F)
    k4 = f_nonlinear(y + dt*k3, F)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0

#pa la app 
pygame.init()
WIDTH, HEIGHT = 1100, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Péndulo Invertido - Control en Cascada PD")
font = pygame.font.SysFont("Arial", 18)
clock = pygame.time.Clock()

# drawing scales
scale = 200  # pixeles por metro
cart_y_px = HEIGHT // 2

# button
run_rect = pygame.Rect(30, 20, 100, 40)
reset_rect = pygame.Rect(150, 20, 100, 40)

simulating = False

# helper: draw text
def draw_text(surf, text, pos, color=(0,0,0)):
    surf.blit(font.render(text, True, color), pos)

# -----------------------
# Main loop
# -----------------------
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = event.pos
            if run_rect.collidepoint(mx, my):
                simulating = True
            elif reset_rect.collidepoint(mx, my):
                # reiniciar estado
                state = initial_state()
                integral_theta = 0.0
                simulating = False

    screen.fill((255,255,255))

    # Draw buttons
    pygame.draw.rect(screen, (0, 150, 0), run_rect)
    draw_text(screen, "RUN", (run_rect.x+32, run_rect.y+10), (255,255,255))
    pygame.draw.rect(screen, (150, 0, 0), reset_rect)
    draw_text(screen, "RESET", (reset_rect.x+18, reset_rect.y+10), (255,255,255))

    # Draw info panel
    draw_text(screen, f"Kp_theta={Kp_theta:.1f}  Kd_theta={Kd_theta:.1f}", (30,80))
    draw_text(screen, f"Kp_pos={Kp_pos:.2f}  Kd_pos={Kd_pos:.2f}", (30,100))
    draw_text(screen, f"force_limit={force_limit:.1f} N", (30,120))

    if simulating:
        # read current states
        x, x_dot, theta, theta_dot = state

        # --- Lazo externo: posición -> theta_ref (PD) ---
        pos_err = 0.0 - x
        theta_ref = Kp_pos * pos_err - Kd_pos * x_dot
        # saturar theta_ref
        if theta_ref > theta_ref_limit: theta_ref = theta_ref_limit
        if theta_ref < -theta_ref_limit: theta_ref = -theta_ref_limit

        # --- Lazo interno: PD sobre theta (puedes activar Ki_theta si quieres) ---
        err_theta = theta_ref - theta
        derivative_theta = -theta_dot  # approx de d(err)/dt
        # integral anti-windup (simple) -- solo si Ki_theta > 0
        if Ki_theta != 0:
            integral_theta += err_theta * dt

        F_unsat = Kp_theta * err_theta + Ki_theta * integral_theta + Kd_theta * derivative_theta

        # saturación de fuerza y anti-windup
        F = max(-force_limit, min(force_limit, F_unsat))
        if Ki_theta != 0:
            if abs(F_unsat) > force_limit:
                # evitar crecer la integral si saturado
                integral_theta -= err_theta * dt

        # protección de límites espaciales
        if state[0] <= -x_limit and F < 0: F = 0.0
        if state[0] >=  x_limit and F > 0: F = 0.0

        # Integrar con RK4
        state = rk4_step(state, F, dt)

        # Draw cart and pendulum
        cart_x_px = int(WIDTH//2 + state[0]*scale)
        cart_y = cart_y_px
        # ensure ints for pygame
        cart_x_px_i = int(cart_x_px)
        cart_y_i = int(cart_y)

        pygame.draw.rect(screen, (0,0,0), (cart_x_px_i - 40, cart_y_i - 20, 80, 40), 2)

        pend_x = cart_x_px + int(l * scale * math.sin(state[2]))
        pend_y = cart_y - int(l * scale * math.cos(state[2]))
        pend_x_i = int(pend_x); pend_y_i = int(pend_y)

        pygame.draw.line(screen, (200,0,0), (cart_x_px_i, cart_y_i), (pend_x_i, pend_y_i), 5)
        pygame.draw.circle(screen, (0,0,200), (pend_x_i, pend_y_i), 10)

        # overlay some values
        draw_text(screen, f"x={state[0]:.3f} m", (30,160))
        draw_text(screen, f"theta={math.degrees(state[2]):.2f} deg", (30,180))
        draw_text(screen, f"F={F:.2f} N", (30,200))

    else:
        # show initial pendulum not simulating
        cart_x_px = int(WIDTH//2 + state[0]*scale)
        cart_y = cart_y_px
        cart_x_px_i = int(cart_x_px); cart_y_i = int(cart_y)
        pygame.draw.rect(screen, (0,0,0), (cart_x_px_i - 40, cart_y_i - 20, 80, 40), 2)
        pend_x = cart_x_px + int(l * scale * math.sin(state[2]))
        pend_y = cart_y - int(l * scale * math.cos(state[2]))
        pend_x_i = int(pend_x); pend_y_i = int(pend_y)
        pygame.draw.line(screen, (200,0,0), (cart_x_px_i, cart_y_i), (pend_x_i, pend_y_i), 5)
        pygame.draw.circle(screen, (0,0,200), (pend_x_i, pend_y_i), 10)
        draw_text(screen, "Presiona RUN para iniciar", (30,240), (80,80,80))

    pygame.display.flip()
    clock.tick(int(1.0/dt))

pygame.quit()
