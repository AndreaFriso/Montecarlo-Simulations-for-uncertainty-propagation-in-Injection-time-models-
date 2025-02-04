import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

# Friction model parameters
viscosity = 0.0036  # [Ns/m2]
r_barrel = 0.00318  # [m]
l_stop = 0.007  # [m]
thick_nominal = 1.5e-9  # [m]
needle_length = 0.02  # [m]
needle_radius = 0.00025  # [m]

# Monte Carlo bounds for thick
thick_lower = thick_nominal * 0.8  # -20%
thick_upper = thick_nominal * 1.2  # +20%

def friction_mechanistic_model(speed, thick):
    """ Mechanistic model for friction force. """
    return ((2 * math.pi * viscosity * r_barrel * l_stop) / thick) * speed

def hydrodynamic_force(speed):
    """ Mechanistic model for hydrodynamic force. """
    radius_barrel = r_barrel
    prod_viscosity = viscosity
    return (((8 * math.pi * prod_viscosity * needle_length * radius_barrel**4) / (needle_radius**4)) * speed)

def piston_simulation_with_injection_time(m, k, x0, F_hydro_func, F_friction_func, x_init, v_init, x_final, t_span, thick):
    """ Simulates the piston motion and calculates the injection time. """
    def equations(t, y):
        x, v = y
        F_spring = -k * (x - x0)
        F_hydro = F_hydro_func(v)
        F_friction = F_friction_func(v, thick)
        F_net = F_spring - F_hydro - F_friction
        a = F_net / m  # Acceleration
        return [v, a]

    # Solve the ODE
    solution = solve_ivp(
        equations, t_span, [x_init, v_init], method='RK45', max_step=0.01, events=lambda t, y: y[0] - x_final
    )

    # Extract the injection time from the event
    if solution.t_events[0].size > 0:
        injection_time = solution.t_events[0][0]  # Time at which x_final is reached
    else:
        injection_time = None  # If x_final is never reached in t_span

    return injection_time

# Parameters
m = 0.05       # Mass of piston [kg]
x0 = 0.0       # Rest position of the spring [m]
x_init = 0.034 # Initial position of the piston [m]
x_final = 0.0  # Final position of the piston [m]
v_init = 0.0   # Initial velocity [m/s]
t_span = (0, 80.0)  # Extended time span [s]

# Range of spring constants to test
spring_constants = np.linspace(100, 1000, 20)  # Test 20 values between 100 and 1000 N/m

# Monte Carlo settings
num_simulations = 10  # Numero di simulazioni per ogni costante elastica

# Dizionario per salvare i risultati delle simulazioni Monte Carlo
monte_carlo_results = {}

for k in spring_constants:
    injection_times_mc = []

    for _ in range(num_simulations):
        # Seleziona un valore casuale di thick entro ±20%
        thick_sample = np.random.uniform(thick_lower, thick_upper)

        # Esegui la simulazione con il valore casuale di thick
        injection_time = piston_simulation_with_injection_time(
            m, k, x0, hydrodynamic_force, friction_mechanistic_model, x_init, v_init, x_final, t_span, thick_sample
        )

        if injection_time is not None:
            injection_times_mc.append(injection_time)

    # Calcola media e deviazione standard
    mean_time = np.mean(injection_times_mc)
    std_time = np.std(injection_times_mc)

    monte_carlo_results[k] = (mean_time, std_time)

# Estrarre i dati per il plot
spring_constants_list = list(monte_carlo_results.keys())
mean_injection_times = [monte_carlo_results[k][0] for k in spring_constants_list]
std_injection_times = [monte_carlo_results[k][1] for k in spring_constants_list]

# Plot della variazione del tempo di iniezione con incertezza
plt.figure(figsize=(10, 6))
plt.errorbar(spring_constants_list, mean_injection_times, yerr=std_injection_times, fmt='o-', capsize=5, label="Mean ± Std Dev")
plt.xlabel("Spring Constant (N/m)")
plt.ylabel("Injection Time (s)")
plt.title("Injection Time Variation with Monte Carlo Simulation on Thick Parameter")
plt.grid()
plt.legend()
plt.show()

# Stampa i risultati
for k, (mean_t, std_t) in monte_carlo_results.items():
    print(f"Spring Constant: {k:.2f} N/m, Mean Injection Time: {mean_t:.2f} s, Std Dev: {std_t:.2f} s")

