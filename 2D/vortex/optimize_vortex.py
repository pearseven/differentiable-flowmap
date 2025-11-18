from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
from mgpcg import *
from init_conditions import *
from io_utils import *
from flowmap import *
from lfm_midpoint_diff_simulator import *
#from neural_buffer import *
import sys
from scipy.optimize import minimize

@ti.data_oriented
class OptimizerSimple:
    def __init__(self):        
        np.random.seed(42)
        self.theta = []
        for i in range(16):
            self.theta.append((0.4*np.random.rand()-0.2)* 5.) 

        positions = []

        centers = [
            ti.Vector([0.75, 0.45]),
            ti.Vector([0.75, 0.15]),
            ti.Vector([0.25, 0.85]),
            ti.Vector([0.25, 0.55]),
            ti.Vector([0.15, 0.75]),
            ti.Vector([0.15, 0.65]),
            ti.Vector([0.85, 0.4]) ,
            ti.Vector([0.85, 0.2]) 
        ]
        # random init around (-0.2-0.2) range
        for c in centers:
            for _ in range(2):
                offset = 0.2 * (np.random.rand(2) - 0.5)
                pos = c + ti.Vector(offset)
                positions.append(pos)

        for pos in positions:
            self.theta.append(pos[0])

        for pos in positions:
            self.theta.append(pos[1])

        for i in range(16):
            self.theta.append(0.2*np.random.rand()+0.5)

        print("self.theta",len(self.theta))
        print("self.theta",self.theta)
        self.target_theta = 0.5
        self.opt_iter = 0
        
        self.alpha1 = 3e-4          #3e-4/2#1e-3#3e-4
        self.alpha2 = 3e-5*2         #3e-5/2#3e-5 ****
        self.alpha3 = 3e-3               #3e-3/2#3e-4

        # at the begging, no /2
        #self.alpha1 = 3e-5/2              #3e-4/2#1e-3#3e-4
        #self.alpha2 = 3e-4/2            #3e-5/2#3e-5 ****
        #self.alpha3 = 3e-3/2               #3e-3/2#3e-4
        #self.alpha1 = 5e-6#1e-3#3e-4
        #self.alpha2 = 3e-6#3e-5 ****
        #self.alpha3 = 3e-4#3e-4
        
        #if(self.opt_iter>20):
        #    self.alpha1 /= 2            
        #    self.alpha2 /= 2            
        #    self.alpha3 /= 2
        
        #if(self.opt_iter>60):
        #    self.alpha1 /= 2            
        #    self.alpha2 /= 2            
        #    self.alpha3 /= 2

        self.history_theta = [self.theta]
        self.history_gradient = []
        self.history_loss = []
        logsdir = os.path.join('logs', exp_name)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        
        log_target_dir = os.path.join('logs', exp_name+"_target")
        save_u_target_dir = "save_u"
        save_u_target_dir = os.path.join(log_target_dir, save_u_target_dir)
        self.save_u_target_dir = save_u_target_dir
        self.simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir,save_u_target_dir)
        self.use_lbfgsb = True 
        self.lbfgsb_state = None  
        self.lbfgsb_iter = 0  

    def init_lbfgsb(self):
        self.lbfgsb_state = {
            'n': len(self.theta),
            'x': np.array(self.theta).copy(),
            'f': 0.0,
            'g': np.zeros(len(self.theta)),
            'H': np.eye(len(self.theta)),  # Initial Hessian approximation
            'nfev': 0,
            'njev': 0,
            'nit': 0,
            'status': 0,
            'message': '',
            'warnflag': 0,
            'task': b'START',
            'old_d': np.zeros(len(self.theta)),
            'rho': [],
            's': [],
            'y': [],
            'alpha': [],
            'sk': []
        }
        self.m = 10 
        for i in range(self.m):
            self.lbfgsb_state['rho'].append(0.0)
            self.lbfgsb_state['s'].append(np.zeros(len(self.theta)))
            self.lbfgsb_state['y'].append(np.zeros(len(self.theta)))
            self.lbfgsb_state['alpha'].append(0.0)
            self.lbfgsb_state['sk'].append(0.0)

    def apply_regularization(self, reg_coef=0.001):
        strengths = self.theta[48:64]
        
        reg_term = reg_coef * np.sum(np.square(strengths))
        
        if hasattr(self.simulator, 'loss'):
            self.simulator.loss += reg_term
        
        return reg_term


    def schedule_lr(self):
        self.alpha1 = 1e-4 / 3.           #3e-4/2#1e-3#3e-4
        self.alpha2 = 1e-5*2 /3.          #3e-5/2#3e-5 ****
        self.alpha3 = 1e-3/3.               #3e-3/2#3e-4
        if(self.opt_iter>=100):
            self.alpha1 /= 2            #3e-4/2#1e-3#3e-4
            self.alpha2 /= 2           #3e-5/2#3e-5 ****
            self.alpha3 /= 2               #3e-3/2#3e-4          
    
    def adjust_lr_by_gradient(self, gradient):
        max_grad1 = np.max(np.abs(gradient[0:16]))
        max_grad2 = np.max(np.abs(gradient[16:48]))
        max_grad3 = np.max(np.abs(gradient[48:64]))
        
        base_alpha1 = 1e-3 
        base_alpha2 = 1e-5
        base_alpha3 = 1e-3
        
        scale_factor = 0.003  # Adjust this to control sensitivity
        self.alpha1 = base_alpha1 / (1.0 + scale_factor * max_grad1)
        self.alpha2 = base_alpha2 / (1.0 + scale_factor * max_grad2)
        self.alpha3 = base_alpha3 / (1.0 + scale_factor * max_grad3)
        
        min_lr = 1e-7
        self.alpha1 = max(self.alpha1, min_lr)
        self.alpha2 = max(self.alpha2, min_lr)
        self.alpha3 = max(self.alpha3, min_lr)
        if self.opt_iter >= 40:
            self.alpha1 /= 2
            self.alpha2 /= 2
            self.alpha3 /= 2
        
        # if self.opt_iter >= 60:
        #     self.alpha1 /= 2
        #     self.alpha2 /= 2
        #     self.alpha3 /= 2
        
        # if self.opt_iter >= 100:
        #     self.alpha1 /= 2
        #     self.alpha2 /= 2
        #     self.alpha3 /= 2
        
        if self.opt_iter >= sub_iters:
            self.alpha1 /= 3
            self.alpha2 /= 3
            self.alpha3 /= 3
        
        if self.opt_iter >= sub_iters * 2:
            self.alpha1 /= 3
            self.alpha2 /= 3
            self.alpha3 /= 3
            
        if self.opt_iter >= sub_iters * 3:
            self.alpha1 /= 3
            self.alpha2 /= 3
            self.alpha3 /= 3
            
        print(f"Dynamic LR: alpha1={self.alpha1:.2e}, alpha2={self.alpha2:.2e}, alpha3={self.alpha3:.2e}")
        
        return self.alpha1, self.alpha2, self.alpha3

    def calculate_dtheta_ind(self, ind, theta,simulator:LFM_Diff_Simulator):
        new_theta = []
        epsilon = 1e-5
        for i in range(64):
            if i == ind:
                new_theta.append(theta[i]+epsilon)
            else:
                new_theta.append(theta[i])
        sixteen_vortex_vel_func_with_pos_coef(simulator.u,simulator.X,theta)
        split_central_vector(simulator.u,simulator.partial_u_x,simulator.partial_u_y)
        # simulator.solver.Poisson(simulator.partial_u_x, simulator.partial_u_y)
        u_x1 = simulator.partial_u_x.to_numpy()
        u_y1 = simulator.partial_u_y.to_numpy()      

        sixteen_vortex_vel_func_with_pos_coef(simulator.u,simulator.X,new_theta)
        split_central_vector(simulator.u,simulator.partial_u_x,simulator.partial_u_y)
        # simulator.solver.Poisson(simulator.partial_u_x, simulator.partial_u_y)
        u_x2 = simulator.partial_u_x.to_numpy()
        u_y2 = simulator.partial_u_y.to_numpy()       

        u_x2 = (u_x2-u_x1)/epsilon
        u_y2 = (u_y2-u_y1)/epsilon

        u_x = simulator.adj_u_x.to_numpy()
        u_y = simulator.adj_u_y.to_numpy()
        
        dtheta = float(np.sum(u_x*u_x2)+np.sum(u_y*u_y2))
        return dtheta 

    def calculate_dtheta(self, simulator):
        theta_list = []
        for i in range(64):
            theta_list.append(self.calculate_dtheta_ind(i, self.theta, simulator))
        return theta_list

    def update_with_lbfgsb(self, simulator):
        if self.lbfgsb_state is None:
            self.init_lbfgsb()
        
        gradient = np.array(self.calculate_dtheta(simulator))
        
        # Store old theta values
        old_theta = np.array(self.theta)
        
        # Get current loss
        current_loss = simulator.loss if hasattr(simulator, 'loss') else 0.0
        
        # Apply dynamic learning rate adjustment
        self.adjust_lr_by_gradient(gradient)
        
        # Create scaling factors for parameter groups
        scales = np.ones(len(self.theta))
        scales[0:16] = self.alpha1
        scales[16:48] = self.alpha2
        scales[48:64] = self.alpha3
        
        # Scale the gradient
        scaled_grad = scales * gradient
        
        # Determine trust region size (can be adjusted based on learning rate)
        trust_region_size = 0.5
        if self.opt_iter >= 60:
            trust_region_size = 0.25
        if self.opt_iter >= 100:
            trust_region_size = 0.125
        if self.opt_iter >= 140:
            trust_region_size = 0.125/2.
        if self.opt_iter >= 180:
            trust_region_size = 0.125/2.
        
        # Define the quadratic model function for L-BFGS-B
        def func(x):
            delta_x = x - old_theta
            
            # Linear approximation
            linear_term = np.dot(scaled_grad, delta_x)
            
            # Quadratic regularization
            quad_coef = 0.5 / trust_region_size
            quadratic_term = quad_coef * np.sum(delta_x**2)
            
            # Model of the loss function
            approx_loss = current_loss + linear_term + quadratic_term + self.apply_regularization()
            
            return approx_loss, scaled_grad
        
        # Set appropriate bounds for parameters
        # First 16 params (0-15): No bounds (vortex initial conditions)
        # Middle 32 params (16-47): Position params bounded in [0,1]
        # Last 16 params (48-63): No bounds (vortex strengths)
        bounds = []
        
        # Add bounds for first 16 params (no bounds)
        for i in range(16):
            bounds.append((None, None))
        
        # Add bounds for position params
        for i in range(16, 48):
            bounds.append((0.0, 1.0))
        
        # Add bounds for strength params (no bounds)
        for i in range(48, 64):
            bounds.append((None, None))
        
        # Import scipy's L-BFGS-B implementation
        from scipy.optimize import fmin_l_bfgs_b
        
        # Run one step of L-BFGS-B
        new_x, f, d = fmin_l_bfgs_b(
            func,
            old_theta,
            bounds=bounds,
            maxiter=1,
            maxfun=5,
            m=10,
            factr=1e7,
            pgtol=1e-3
        )
        
        # Update theta with optimized values
        self.theta = new_x.tolist()
        
        # Update L-BFGS-B memory
        if self.lbfgsb_iter > 0:
            s = new_x - old_theta
            y = gradient - self.lbfgsb_state['g']
            
            # Only update if s and y satisfy curvature condition
            ys = np.dot(y, s)
            if ys > 1e-10:
                idx = self.lbfgsb_iter % self.m
                self.lbfgsb_state['s'][idx] = s
                self.lbfgsb_state['y'][idx] = y
                self.lbfgsb_state['rho'][idx] = 1.0 / ys
        
        # Update state
        self.lbfgsb_state['x'] = new_x
        self.lbfgsb_state['g'] = gradient
        self.lbfgsb_iter += 1
        
        print(f"LBFGS-B update complete, iteration {self.lbfgsb_iter}")
        
        return gradient.tolist()

    def calc_target(self):
        logsdir = os.path.join('logs', exp_name+"_target")
        os.makedirs(logsdir, exist_ok=True)
        remove_everything_in(logsdir)
        vortdir = 'vorticity'
        vortdir = os.path.join(logsdir, vortdir)
        os.makedirs(vortdir, exist_ok=True)
        passive1dir = 'passive1'
        passive1dir = os.path.join(logsdir, passive1dir)
        os.makedirs(passive1dir, exist_ok=True)
        passive2dir = 'passive2'
        passive2dir = os.path.join(logsdir, passive2dir)
        os.makedirs(passive2dir, exist_ok=True)
        passive3dir = 'passive3'
        passive3dir = os.path.join(logsdir, passive3dir)
        os.makedirs(passive3dir, exist_ok=True)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        os.makedirs(save_u_dir, exist_ok=True)
        shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')
        simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir)
        simulator.init([-0.6,0.6,-0.6,0.6],target= True)    
        w_numpy = simulator.calculate_w()
        w_max = 15
        w_min = -15
        write_field(w_numpy, vortdir, 0 , vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
        last_output_substep = 0
        while True:
            output_frame, final_flag = simulator.forward_step_midpoint(True, True)
            if output_frame:
                w_numpy = simulator.calculate_w()     
                write_field(w_numpy, vortdir, simulator.frame_idx[None], vmin=w_min, vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    passive_numpy = simulator.passive1.to_numpy()
                    write_field(passive_numpy, passive1dir, simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                    passive_numpy = simulator.passive2.to_numpy()
                    write_field(passive_numpy, passive2dir, simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                    passive_numpy = simulator.passive3.to_numpy()
                    write_field(passive_numpy, passive3dir, simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                print("[Simulate] Finished frame: ", simulator.frame_idx[None], " in ", simulator.step_num[None]-last_output_substep, "substeps \n\n")
                last_output_substep = simulator.step_num[None]
            if final_flag:
                break

    def iter(self):
        self.opt_iter+=1 
        begin_step = 0
        end_step = total_steps
        if sub_optimize == True:
            end_step = sub_steps * (self.opt_iter // sub_iters + 1)
        self.schedule_lr()
        logsdir = os.path.join('logs', exp_name)
        os.makedirs(logsdir, exist_ok=True)
        remove_everything_in(logsdir)
        vortdir = 'vorticity'
        vortdir = os.path.join(logsdir, vortdir)
        os.makedirs(vortdir, exist_ok=True)
        passive1dir = 'passive1'
        passive1dir = os.path.join(logsdir, passive1dir)
        os.makedirs(passive1dir, exist_ok=True)
        passive2dir = 'passive2'
        passive2dir = os.path.join(logsdir, passive2dir)
        os.makedirs(passive2dir, exist_ok=True)
        passive3dir = 'passive3'
        passive3dir = os.path.join(logsdir, passive3dir)
        os.makedirs(passive3dir, exist_ok=True)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        os.makedirs(save_u_dir, exist_ok=True)
        shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')
        #self.simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir,self.save_u_target_dir)
        self.simulator.init(self.theta)    
        w_numpy = self.simulator.calculate_w()
        w_max = 15
        w_min = -15
        write_field(w_numpy, vortdir, 0, vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
        if(add_passive_scalar):
            passive_numpy = self.simulator.passive1.to_numpy()
            write_field(passive_numpy, passive1dir, 0, vmin=0, vmax=1, dpi=dpi_vor)
            passive_numpy = self.simulator.passive2.to_numpy()
            write_field(passive_numpy, passive2dir, 0, vmin=0, vmax=1, dpi=dpi_vor)
            passive_numpy = self.simulator.passive3.to_numpy()
            write_field(passive_numpy, passive3dir, 0, vmin=0, vmax=1, dpi=dpi_vor)
        last_output_substep = 0
        # Forward Step
        # self.simulator.step_num[None] = begin_step
        while True:
            if self.simulator.step_num[None] < end_step:
                print(begin_step, end_step)
                output_frame, final_flag = self.simulator.forward_step_midpoint(True,True)
                if output_frame:
                    w_numpy = self.simulator.calculate_w()     
                    write_field(w_numpy, vortdir, self.simulator.frame_idx[None], vmin=w_min,
                        vmax=w_max, dpi=dpi_vor)
                    if(add_passive_scalar):
                        passive_numpy = self.simulator.passive1.to_numpy()
                        write_field(passive_numpy, passive1dir, self.simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                        passive_numpy = self.simulator.passive2.to_numpy()
                        write_field(passive_numpy, passive2dir, self.simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                        passive_numpy = self.simulator.passive3.to_numpy()
                        write_field(passive_numpy, passive3dir, self.simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                    print("[Simulate] Finished frame: ", self.simulator.frame_idx[None], " in ", self.simulator.step_num[None]-last_output_substep, "substeps \n\n")
                    last_output_substep = self.simulator.step_num[None]
                if final_flag:
                    break
            else:
                break
        
        logsdir = os.path.join('logs', exp_name)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        #self.simulator = LFM_Diff_self.simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir)
        #self.simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir,self.save_u_target_dir)
        self.simulator.init_gradient()
        gvortdir = 'gradient_vorticity'
        gvortdir = os.path.join(logsdir, gvortdir)
        os.makedirs(gvortdir, exist_ok=True)
        gpassive1dir = 'gradient_passive1'
        gpassive1dir = os.path.join(logsdir, gpassive1dir)
        os.makedirs(gpassive1dir, exist_ok=True)
        gpassive2dir = 'gradient_passive2'
        gpassive2dir = os.path.join(logsdir, gpassive2dir)
        os.makedirs(gpassive2dir, exist_ok=True)
        gpassive3dir = 'gradient_passive3'
        gpassive3dir = os.path.join(logsdir, gpassive3dir)
        os.makedirs(gpassive3dir, exist_ok=True)
        w_max = 1.5#15*3
        w_min = -1.5#-15*3
        
        # Backward Step
        while True:
            output_frame, final_flag = self.simulator.backtrack_step_midpoint(False,False)
            if output_frame:
                w_numpy = self.simulator.calculate_gw()     
                write_field(w_numpy, gvortdir, self.simulator.frame_idx[None], vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    write_field(self.simulator.adj_passive1.to_numpy(),gpassive1dir, self.simulator.frame_idx[None], vmin=0,
                        vmax=1, dpi=dpi_vor)
                    write_field(self.simulator.adj_passive2.to_numpy(),gpassive2dir, self.simulator.frame_idx[None], vmin=0,
                        vmax=1, dpi=dpi_vor)
                    write_field(self.simulator.adj_passive3.to_numpy(),gpassive3dir, self.simulator.frame_idx[None], vmin=0,
                        vmax=1, dpi=dpi_vor)
            if final_flag:
                break
        gradient = self.calculate_dtheta(self.simulator)
        if self.use_lbfgsb:
            gradient = self.update_with_lbfgsb(self.simulator)
            print("Using Newton!")
        else:
            gradient = self.calculate_dtheta(self.simulator)
            for i in range(16):
                self.theta[i] = self.theta[i] - self.alpha1*gradient[i]
            for i in range(16,48):
                self.theta[i] = self.theta[i] - self.alpha2*gradient[i]
            for i in range(48,64):
                self.theta[i] = self.theta[i] - self.alpha3*gradient[i]

        self.history_theta.append(self.theta)
        self.history_gradient.append(gradient)
        self.history_loss.append(self.simulator.loss)

def write_file(l,file):
    with open(file, "a") as f:
        for item in l:
            f.write(str(item) + ",")
        f.write("\n")

if __name__ == '__main__':
    opt = OptimizerSimple()
    opt.opt_iter = 0
    # simulate_target = True
    # if(simulate_target):
    opt.calc_target()
    # else:         
    log_file = "./16_vortex_for_8_vortex_opt_log.txt" 
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Deleted: {log_file}")
    else:
        print(f"File does not exist: {log_file}")
    write_file([f"iter:{opt.opt_iter}-"+"[loss:]"]+opt.history_loss,log_file)
    write_file([f"iter:{opt.opt_iter}-"+"[theta:]"]+opt.history_theta[-1],log_file)
    write_file([f"iter:{opt.opt_iter}-"+"[gradient:]"]+[],log_file)
    while(True):
        opt.iter()
        write_file([f"iter:{opt.opt_iter}-"+"[loss:]"]+opt.history_loss,log_file)
        write_file([f"iter:{opt.opt_iter}-"+"[theta:]"]+opt.history_theta[-1],log_file)
        write_file([f"iter:{opt.opt_iter}-"+"[gradient:]"]+opt.history_gradient[-1],log_file)