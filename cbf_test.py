import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import math

def forward_dynamics_opt(timestep):
    """Return updated state in a form of `ca.SX`"""
    l = 3 #here is wheelbase
    x_symbol = ca.SX.sym("x", 4)
    u_symbol = ca.SX.sym("u", 2)
    x_symbol_next = x_symbol[0] + x_symbol[2] * ca.cos(x_symbol[3]) * timestep
    y_symbol_next = x_symbol[1] + x_symbol[2] * ca.sin(x_symbol[3]) * timestep
    v_symbol_next = x_symbol[2] + u_symbol[0] * timestep
    theta_symbol_next = x_symbol[3] + x_symbol[2] * ca.tan(u_symbol[1]) / l * timestep
    state_symbol_next = ca.vertcat(x_symbol_next, y_symbol_next, v_symbol_next, theta_symbol_next)
    return ca.Function("dubin_car_dynamics", [x_symbol, u_symbol], [state_symbol_next])

def forward_dynamics(x, u, timestep):
        """Return updated state in a form of `np.ndnumpy`"""
        l = 3
        x_next = np.ndarray(shape=(4,), dtype=float)
        x_next[0] = x[0] + x[2] * math.cos(x[3]) * timestep
        x_next[1] = x[1] + x[2] * math.sin(x[3]) * timestep
        x_next[2] = x[2] + u[0] * timestep
        x_next[3] = x[3] + x[2] * math.tan(u[1]) / l * timestep
        return x_next

class cbf:
    def __init__(self):
        self.horizon = 8
        self.mat_Q = np.diag([10.0, 10.0, 10.0, 10.0])
        self.mat_P = np.diag([100.0, 100.0, 100.0, 100.0])
        self.mat_R = np.diag([1.0, 1.0])
        self.cbf_gamma = 0.5

        self.opti = ca.Opti()
        self.dynamics_opt = forward_dynamics_opt(0.1)

        self.variables = dict()

        # set your obstacle
        self.obs_x = -2
        self.obs_y = -2.25
        self.obs_r = 1.5
    
    def solve_cbf(self, xk):
        self.variables["x"] = self.opti.variable(4, self.horizon + 1)
        self.variables["u"] = self.opti.variable(2, self.horizon)
        cost = 0

        # init constrain
        self.opti.subject_to(self.variables["x"][:, 0] == xk)
        # bound cons
        amin, amax = -0.5, 0.5
        omegamin, omegamax = -0.6, 0.6
        for i in range(self.horizon):
            self.opti.subject_to(self.variables["u"][0, i] <= amax)
            self.opti.subject_to(amin <= self.variables["u"][0, i])
            self.opti.subject_to(self.variables["u"][1, i] <= omegamax)
            self.opti.subject_to(omegamin <= self.variables["u"][1, i])
        # kinematic cons
        for i in range(self.horizon):
            self.opti.subject_to(
                self.variables["x"][:, i + 1] == self.dynamics_opt(self.variables["x"][:, i], self.variables["u"][:, i])
            )
        # mpc_cost
        for i in range(self.horizon):
            cost += ca.mtimes(ca.mtimes(self.variables["x"][:, i].T, self.mat_Q), self.variables["x"][:, i])
            cost += ca.mtimes(ca.mtimes(self.variables["u"][:, i].T, self.mat_R), self.variables["u"][:, i])
        # terminal cons
        cost += ca.mtimes(ca.mtimes(self.variables["x"][:, -1].T, self.mat_P), self.variables["x"][:, -1])
        # cbf_cons
        obstacle = [self.obs_x, self.obs_y]
        for i in range(self.horizon - 1):
            x = ca.mtimes((self.variables["x"][:2, i]  - obstacle).T, self.variables["x"][:2, i]  - obstacle) - (self.obs_r)**2
            x_next = ca.mtimes((self.variables["x"][:2, i+1]  - obstacle).T, self.variables["x"][:2, i+1]  - obstacle) - (self.obs_r)**2
            self.opti.subject_to(x_next - x + self.cbf_gamma * x >= 0) 
        
        self.opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", option)
        try:
            opt_sol = self.opti.solve()
            return opt_sol.value(self.variables["u"][:, 0])
        except Exception as e:
            print(f"Solver failed: {e}")
            assert False

def plot_gradient_circle(x, y, r):
    theta = np.linspace(0, 2*np.pi, 1000)
    x_points = x + r * np.cos(theta)
    y_points = y + r * np.sin(theta)
    
    plt.plot(x_points, y_points)


cbf = cbf()
xk = [-5, -5, np.pi/4, 0]

plot_gradient_circle(cbf.obs_x, cbf.obs_y, cbf.obs_r)

x = []
y = []
for i in range(50):
    if xk[0]**2 + xk[1]**2 <= 0.1: break
    x.append(xk[0])
    y.append(xk[1])
    u = cbf.solve_cbf(xk)
    xk = forward_dynamics(xk, u, 0.1)
plt.gca().set_aspect('equal')
plt.plot(x,y)
plt.show()







