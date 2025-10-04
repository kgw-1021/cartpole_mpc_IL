# mpc.py
import casadi as ca
import numpy as np

class CartPoleMPC:
    def __init__(self, horizon=20, dt=0.02, solver_type='osqp'):
        self.solver_type = solver_type
        self.N = int(horizon)
        self.dt = float(dt)
        self._build_ocp()

    def _build_ocp(self):
        N = self.N
        dt = self.dt

        # dimensions
        nx = 4
        nu = 1

        x_sym = ca.SX.sym('x_sym', nx)   # [x, x_dot, theta, theta_dot]
        u_sym = ca.SX.sym('u_sym', nu)   # [u]

        x_pos = x_sym[0]
        x_dot = x_sym[1]
        theta = x_sym[2]
        theta_dot = x_sym[3]
        u = u_sym[0]

        # physical parameters
        g = 9.81
        m_cart = 1.0
        m_pole = 0.1
        l = 0.5
        total_mass = m_cart + m_pole
        polemass_length = m_pole * l

        # dynamics
        temp = (u + polemass_length * theta_dot**2 * ca.sin(theta)) / total_mass
        theta_acc = (g * ca.sin(theta) - ca.cos(theta) * temp) / \
                    (l * (4.0/3.0 - m_pole * ca.cos(theta)**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * ca.cos(theta) / total_mass

        f_cont = ca.vertcat(
            x_dot,
            x_acc,
            theta_dot,
            theta_acc
        )

        # dynamics integration (forward Euler)
        x_next_expr = x_sym + dt * f_cont
        f_disc = ca.Function('f_disc', [x_sym, u_sym], [x_next_expr])

        # variables for OCP
        X = ca.SX.sym('X', nx, N + 1)   # states over horizon (0..N)
        U = ca.SX.sym('U', nu, N)       # controls over horizon (0..N-1)
        P = ca.SX.sym('P', nx)          # parameter: initial state

        # Weights
        Q = np.diag([1.0, 0.1, 10.0, 0.1])   # state weight
        R = np.array([[0.01]])               # input weight
        Qc = ca.DM(Q)
        Rc = ca.DM(R)

        # cost
        cost = ca.SX(0)
        g = []

        g.append(X[:, 0] - P)

        # roll out
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            # discrete dynamics prediction
            x_next_pred = f_disc(xk, uk)
            # dynamics equality constraint
            g.append(X[:, k + 1] - x_next_pred)
            # stage cost
            cost += ca.mtimes([xk.T, Qc, xk]) + ca.mtimes([uk.T, Rc, uk])

        vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        cons = ca.vertcat(*g)

        # NLP definition
        nlp = {'x': vars, 'f': cost, 'g': cons, 'p': P}

        # IPOPT solver
        if self.solver_type == 'ipopt':
            opts = {
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.max_iter': 200,
            }
            self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # backend OSQP solver
        elif self.solver_type == 'osqp':
            opts = {"print_time": 0, "print_problem" : 0, "print_out" : 0, "verbose": False,     
                "osqp": {
                "verbose": False,
                "eps_abs": 1e-6,
                "eps_rel": 1e-6
                }
            }
            self.solver = ca.qpsol("qp", "osqp", nlp, opts)

        else:
            raise ValueError(f"Unknown solver type: {self.solver_type}")

        self.nx = nx
        self.nu = nu
        self.N = N
        self.vars = vars
        self.X = X
        self.U = U
        self.P = P
        self.cons_dim = cons.size1()  

    def control(self, x0):

        N = self.N
        nx = self.nx
        nu = self.nu

        x0 = np.asarray(x0).reshape(nx,)
        x_init = np.tile(x0.reshape(-1, 1), (1, N + 1)) 
        u_init = np.zeros((nu, N))                       
        var_init = np.concatenate([x_init.flatten(), u_init.flatten()]).astype(float)

        lbg = np.zeros((self.cons_dim,), dtype=float)
        ubg = np.zeros((self.cons_dim,), dtype=float)

        sol = self.solver(x0=var_init, p=x0, lbg=lbg, ubg=ubg)

        sol_vars = sol['x'].full().flatten()
  
        offset = nx * (N + 1)
        u_opt = sol_vars[offset:].reshape(nu, N)

        return float(u_opt[0, 0])
