from controller import Robot
import time
import os

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Simulación de control simple
start_time = time.time()
best_time_file = "best_time.txt"

def load_best_time():
    if os.path.exists(best_time_file):
        with open(best_time_file, "r") as f:
            return float(f.read().strip())
    return 0.0

def save_best_time(t):
    with open(best_time_file, "w") as f:
        f.write(f"{t:.2f}")

best_time = load_best_time()
print(f"🔥 Mejor tiempo registrado: {best_time:.2f} s")

while robot.step(timestep) != -1:
    elapsed = time.time() - start_time
    print(f"⏱ Tiempo: {elapsed:.2f} s", end="\r")
    
    # Aquí los estudiantes implementarán el control del péndulo
    # ----------------------------------------------------------
    # Lectura de sensores y aplicación de fuerza
    # ----------------------------------------------------------
    
    if elapsed > best_time:
        save_best_time(elapsed)
        best_time = elapsed
        print(f"\n🏆 ¡Nuevo récord! {elapsed:.2f} s")
