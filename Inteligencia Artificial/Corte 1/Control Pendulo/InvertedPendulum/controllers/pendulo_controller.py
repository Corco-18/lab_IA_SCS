from controller import Robot
import math

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Dispositivos
car_motor = robot.getDevice("car_motor")
car_sensor = robot.getDevice("car_sensor")
pendulo_motor = robot.getDevice("pendulo_motor")
pendulo_sensor = robot.getDevice("pendulo_sensor")

car_sensor.enable(timestep)
pendulo_sensor.enable(timestep)

car_motor.setPosition(float('inf'))    # Control por velocidad
pendulo_motor.setPosition(float('inf')) # Control por velocidad

car_speed = 0.5
time_elapsed = 0

while robot.step(timestep) != -1:
    time_elapsed += timestep / 1000.0

    # Movimiento simple de ida y vuelta del carro
    target_speed = car_speed * math.sin(time_elapsed)
    car_motor.setVelocity(target_speed)

    # Lectura ángulo péndulo
    pend_angle = pendulo_sensor.getValue()

    # Control simple para el péndulo: torque proporcional negativo para que oscile
    torque = -5.0 * pend_angle
    pendulo_motor.setVelocity(torque)
