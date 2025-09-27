import casadi as ca
import numpy as np

class CartPoleILQR:
    def __init__(self, horizon=20, dt=0.02, max_iters=50, tol=1e-6):
        self.N = int(horizon)
        self.dt = float(dt)
        self.max_iters = max_iters
        self.tol = tol
        self._build_dynamics_and_cost()
        
        # Initialize nominal trajectory
        self.x_nom = None
        self.u_nom = None
        
    def _build_dynamics_and_cost(self):
        N = self.N
        dt = self.dt
        
        # dimensions
        self.nx = 4  # [x, x_dot, theta, theta_dot]
        self.nu = 1  # [u]
        
        # symbolic variables
        x_sym = ca.SX.sym('x', self.nx)
        u_sym = ca.SX.sym('u', self.nu)
        
        # state components
        x_pos = x_sym[0]
        x_dot = x_sym[1] 
        theta = x_sym[2]
        theta_dot = x_sym[3]
        u = u_sym[0]
        
        # physical parameters (match Gym implementation)
        g = 9.81
        m_cart = 1.0
        m_pole = 0.1
        l = 0.5
        total_mass = m_cart + m_pole
        polemass_length = m_pole * l
        
        # continuous-time dynamics
        temp = (u + polemass_length * theta_dot**2 * ca.sin(theta)) / total_mass
        theta_acc = (g * ca.sin(theta) - ca.cos(theta) * temp) / \
                    (l * (4.0/3.0 - m_pole * ca.cos(theta)**2 / total_mass))
        x_acc = temp - polemass_length * theta_acc * ca.cos(theta) / total_mass
        
        f_cont = ca.vertcat(x_dot, x_acc, theta_dot, theta_acc)
        
        # discrete dynamics (forward Euler)
        x_next = x_sym + dt * f_cont
        
        # dynamics function
        self.f = ca.Function('f', [x_sym, u_sym], [x_next])
        
        # jacobians for linearization
        self.fx = ca.Function('fx', [x_sym, u_sym], [ca.jacobian(x_next, x_sym)])
        self.fu = ca.Function('fu', [x_sym, u_sym], [ca.jacobian(x_next, u_sym)])
        
        # cost weights
        Q = np.diag([1.0, 0.1, 10.0, 0.1])   # state weight
        R = np.array([[0.01]])               # input weight
        
        # stage cost function: x'Qx + u'Ru
        stage_cost = ca.mtimes([x_sym.T, Q, x_sym]) + ca.mtimes([u_sym.T, R, u_sym])
        self.l = ca.Function('l', [x_sym, u_sym], [stage_cost])
        
        # cost derivatives (second-order)
        self.lx = ca.Function('lx', [x_sym, u_sym], [ca.jacobian(stage_cost, x_sym).T])
        self.lu = ca.Function('lu', [x_sym, u_sym], [ca.jacobian(stage_cost, u_sym).T])
        self.lxx = ca.Function('lxx', [x_sym, u_sym], [ca.hessian(stage_cost, x_sym)[0]])
        self.luu = ca.Function('luu', [x_sym, u_sym], [ca.hessian(stage_cost, u_sym)[0]])
        self.lux = ca.Function('lux', [x_sym, u_sym], [ca.jacobian(ca.jacobian(stage_cost, u_sym), x_sym)])
        
        # terminal cost (same as stage cost)
        self.lf = ca.Function('lf', [x_sym], [ca.mtimes([x_sym.T, Q, x_sym])])
        self.lfx = ca.Function('lfx', [x_sym], [ca.jacobian(ca.mtimes([x_sym.T, Q, x_sym]), x_sym).T])
        self.lfxx = ca.Function('lfxx', [x_sym], [ca.hessian(ca.mtimes([x_sym.T, Q, x_sym]), x_sym)[0]])
        
    def _rollout(self, x0, u_seq):
        """Forward rollout given control sequence"""
        x_seq = np.zeros((self.nx, self.N + 1))
        x_seq[:, 0] = x0
        
        for k in range(self.N):
            x_next = self.f(x_seq[:, k], u_seq[:, k]).full().flatten()
            x_seq[:, k + 1] = x_next
            
        return x_seq
    
    def _backward_pass(self, x_seq, u_seq):
        """Backward pass to compute optimal gains"""
        N = self.N
        
        # initialize value function at terminal time
        Vx = self.lfx(x_seq[:, N]).full().flatten()
        Vxx = self.lfxx(x_seq[:, N]).full()
        
        # gains storage
        K = np.zeros((self.nu, self.nx, N))  # feedback gains
        k = np.zeros((self.nu, N))           # feedforward gains
        
        # expected cost reduction
        expected_cost_reduction = 0.0
        
        # backward recursion
        for i in range(N-1, -1, -1):
            x = x_seq[:, i]
            u = u_seq[:, i]
            
            # dynamics jacobians
            fx = self.fx(x, u).full()
            fu = self.fu(x, u).full()
            
            # cost derivatives
            lx = self.lx(x, u).full().flatten()
            lu = self.lu(x, u).full().flatten()
            lxx = self.lxx(x, u).full()
            luu = self.luu(x, u).full()
            lux = self.lux(x, u).full()
            
            # Q-function derivatives
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx
            
            # regularization for numerical stability
            reg = 1e-9
            Quu_reg = Quu + reg * np.eye(self.nu)
            
            # check positive definiteness
            try:
                np.linalg.cholesky(Quu_reg)
            except np.linalg.LinAlgError:
                print(f"Warning: Quu not positive definite at step {i}")
                Quu_reg = Quu + 1e-3 * np.eye(self.nu)
            
            # compute gains
            Quu_inv = np.linalg.inv(Quu_reg)
            K[:, :, i] = -Quu_inv @ Qux
            k[:, i] = -Quu_inv @ Qu
            
            # update value function
            Vx = Qx + K[:, :, i].T @ Quu @ k[:, i] + K[:, :, i].T @ Qu + Qux.T @ k[:, i]
            Vxx = Qxx + K[:, :, i].T @ Quu @ K[:, :, i] + K[:, :, i].T @ Qux + Qux.T @ K[:, :, i]
            
            # expected cost reduction
            expected_cost_reduction += -0.5 * k[:, i].T @ Quu @ k[:, i] - k[:, i].T @ Qu
            
        return K, k, expected_cost_reduction
    
    def _forward_pass(self, x0, x_seq, u_seq, K, k, alpha=1.0):
        """Forward pass with line search"""
        N = self.N
        
        # new sequences
        x_new = np.zeros((self.nx, N + 1))
        u_new = np.zeros((self.nu, N))
        
        x_new[:, 0] = x0
        
        # forward simulation with control update
        for i in range(N):
            # control update
            dx = x_new[:, i] - x_seq[:, i]
            du = alpha * k[:, i] + K[:, :, i] @ dx
            u_new[:, i] = u_seq[:, i] + du
            
            # state propagation
            x_new[:, i + 1] = self.f(x_new[:, i], u_new[:, i]).full().flatten()
        
        return x_new, u_new
    
    def _compute_cost(self, x_seq, u_seq):
        """Compute total cost for trajectory"""
        cost = 0.0
        
        # stage costs
        for k in range(self.N):
            cost += self.l(x_seq[:, k], u_seq[:, k]).full().item()
        
        # terminal cost
        cost += self.lf(x_seq[:, self.N]).full().item()
        
        return cost
    
    def solve(self, x0):
        """Solve ILQR problem"""
        x0 = np.asarray(x0).reshape(-1)
        
        # initialize nominal trajectory if not exists
        if self.x_nom is None or self.u_nom is None:
            self.u_nom = np.zeros((self.nu, self.N))
            self.x_nom = self._rollout(x0, self.u_nom)
        
        # shift previous solution as warm start
        if self.x_nom.shape[1] == self.N + 1:
            # shift states and controls
            self.x_nom = np.roll(self.x_nom, -1, axis=1)
            self.x_nom[:, -1] = self.x_nom[:, -2]  # repeat last state
            self.u_nom = np.roll(self.u_nom, -1, axis=1)
            self.u_nom[:, -1] = 0.0  # zero last control
        
        x_seq = self.x_nom.copy()
        u_seq = self.u_nom.copy()
        
        # ILQR iterations
        for iter in range(self.max_iters):
            # backward pass
            K, k, expected_reduction = self._backward_pass(x_seq, u_seq)
            
            if expected_reduction < self.tol:
                break
            
            # line search in forward pass
            current_cost = self._compute_cost(x_seq, u_seq)
            
            for alpha in [1.0, 0.5, 0.25, 0.1]:
                x_new, u_new = self._forward_pass(x0, x_seq, u_seq, K, k, alpha)
                new_cost = self._compute_cost(x_new, u_new)
                
                if new_cost < current_cost:
                    x_seq = x_new
                    u_seq = u_new
                    break
            else:
                # no improvement found
                break
        
        # store for next solve (warm start)
        self.x_nom = x_seq
        self.u_nom = u_seq
        
        return x_seq, u_seq
    
    def control(self, x0):
        """
        Get optimal control for current state.
        Returns first control input (float).
        """
        x_seq, u_seq = self.solve(x0)
        return float(u_seq[0, 0])