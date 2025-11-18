from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
from mgpcg import *
from init_conditions import *
from io_utils import *
from flowmap import *
from lfm_midpoint_diff_simulator import *
#from neural_buffer import *
import sys,random
from scipy.optimize import fmin_l_bfgs_b

@ti.data_oriented
class OptimizerSimple:
    def __init__(self):        
        np.random.seed(42)
        random.seed(42)
        load_data = False
        if(not load_data):
            self.gaussion_force_num = control_num
            self.gaussion_force_radius_range = [
                1/(dx)**2,
                1/(2*dx)**2,
                1/(3*dx)**2,
                1/(4*dx)**2,
                1/(5*dx)**2,
                1/(6*dx)**2,
                1/(8*dx)**2,
                1/(9*dx)**2,
                1/(10*dx)**2
                # 1/(8*dx)**2
                # 1/(12*dx)**2,
                # 1/(16*dx)**2,
                # 1/(20*dx)**2,
                # 1/(24*dx)**2
            ]
            self.gaussion_force_center = [np.zeros((control_num, 2)) for  i in range(total_steps)]
            self.gaussion_force_radius = [np.zeros((control_num)) for  i in range(total_steps)]
            self.gaussion_force_strength = [np.zeros((control_num, 2)) for  i in range(total_steps)]

            lb_x, ub_x = 0.2,0.8
            lb_y, ub_y = 0.2,0.8
            for j in range(total_steps):
                for i in range(self.gaussion_force_num):
                    self.gaussion_force_radius[j][i] = random.choice(self.gaussion_force_radius_range)
                    self.gaussion_force_center[j][i,0] = random.uniform(lb_x, ub_x)
                    self.gaussion_force_center[j][i,1] = random.uniform(lb_y, ub_y)
                    angle = np.random.uniform(0, 2 * np.pi)
                    magnitude = np.random.uniform(5e-3, 2e-2) * 2.
                    
                    self.gaussion_force_strength[j][i, 0] = magnitude * np.cos(angle)
                    self.gaussion_force_strength[j][i, 1] = magnitude * np.sin(angle)

        else:
            self.gaussion_force_num = control_num
            data = np.load("checkpoint_gaussian_forces_draigon4.npz")
            center = data["center"]
            radius = data["radius"]
            strength = data["strength"]
            self.gaussion_force_center = [center[i,:,:] for  i in range(total_steps)]
            self.gaussion_force_radius = [radius[i,:] for  i in range(total_steps)]
            self.gaussion_force_strength = [strength[i,:,:] for  i in range(total_steps)]
            #for j in range(total_steps):
            #    for i in range(self.gaussion_force_num):
            #        self.gaussion_force_radius[j][i] = self.gaussion_force_radius[j][i]*(1+random.random()*0.01)
            #        self.gaussion_force_center[j][i,0] = self.gaussion_force_center[j][i,0]*(1+random.random()*0.01)
            #        self.gaussion_force_center[j][i,1] = self.gaussion_force_center[j][i,1]*(1+random.random()*0.01)

        self.opt_iter = 0
        self.alpha_center = 1e-5
        self.alpha_strength = 1e-5
        self.alpha_norm_center = 1e-5
        self.alpha_norm_strength = 1e-5

        self.history_theta = []
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

        self.use_lbfgsb = False
        self.lbfgsb_state = None
        self.lbfgsb_iter = 0
    
    def schedule_lr(self):
        self.alpha_center = 2e-5
        self.alpha_strength = 2e-5
        # self.alpha_norm_center = 1e-4*4
        # self.alpha_norm_strength = 1e-5*4
        
        trust_region_size = 0.125
        
        if self.opt_iter > 80:
            self.alpha_center /= 2
            self.alpha_strength /= 2
            self.alpha_norm_center /= 2
            self.alpha_norm_strength /= 2
            trust_region_size = 0.125
            
        if self.opt_iter > 100:
            self.alpha_center /= 1.5
            self.alpha_strength /= 1.5
            self.alpha_norm_center /= 1.5
            self.alpha_norm_strength /= 1.5
            trust_region_size = 0.125/2.
            
        if self.opt_iter > 120:
            self.alpha_center /= 1.5
            self.alpha_strength /= 1.5
            self.alpha_norm_center /= 1.5
            self.alpha_norm_strength /= 1.5
            trust_region_size = 0.125/4.

        if self.opt_iter > 140:
            self.alpha_center /= 1.2
            self.alpha_strength /= 1.2
            self.alpha_norm_center /= 1.5
            self.alpha_norm_strength /= 1.5
            trust_region_size = 0.125 /4.
        
        if len(self.history_loss) >= 5:
            recent_losses = self.history_loss[-5:]
            loss_change_ratio = abs(recent_losses[-1] - recent_losses[0]) / max(abs(recent_losses[0]), 1e-10)
            
            if loss_change_ratio < 0.01:
                self.alpha_center *= 0.9
                self.alpha_strength *= 0.9
                trust_region_size *= 0.9
        
        return trust_region_size
    
    def reposition_weak_forces(self, j):
        # Determine cut percentage based on iteration number
        if self.opt_iter <= 30:
            cut_percent = 0.6  # First time: 50%
        elif self.opt_iter <= 60:
            cut_percent = 0.5  # Second time: 40%
        elif self.opt_iter <= 90:
            cut_percent = 0.4  # Third time: 30%
        elif self.opt_iter <= 120:
            cut_percent = 0.3  # Fourth time: 20%
        elif self.opt_iter <= 150:
            cut_percent = 0.2  # Fifth time: 10%
        else:
            cut_percent = 0.1  # After that: 5%
        
        print(f"Iteration {self.opt_iter}: Repositioning weak forces... Cut percentage: {cut_percent}")
        
        adj_passive_np = self.simulator.adj_passive.to_numpy()

        lb_x, ub_x = 0.0, 1.0
        lb_y, ub_y = 0.0, 1.0
    
        adj_field = np.abs(adj_passive_np)
        
        if np.sum(adj_field) > 0:
            adj_prob = adj_field / np.sum(adj_field)
        else:
            adj_prob = np.ones_like(adj_field) / adj_field.size
        
        flat_prob = adj_prob.flatten()
        
        force_strengths = np.sum(np.abs(self.gaussion_force_strength[j]), axis=1)
        sorted_indices = np.argsort(force_strengths)
        
        cutoff = int(self.gaussion_force_num * cut_percent)
        weak_indices = sorted_indices[:cutoff]
        
        y_coords, x_coords = np.mgrid[0:1:res_y*1j, 0:1:res_x*1j]
        flat_x_coords = x_coords.flatten()
        flat_y_coords = y_coords.flatten()
        
        # Sample positions based on the adjoint field probability
        sampled_indices = np.random.choice(
            range(len(flat_prob)), 
            size=len(weak_indices), 
            p=flat_prob,
            replace=True
        )
        
        for idx_pos, idx_force in enumerate(weak_indices):
            radius_idx = np.random.randint(len(self.gaussion_force_radius_range) // 2)
            
            self.gaussion_force_radius[j][idx_force] = self.gaussion_force_radius_range[radius_idx]
            
            flat_idx = sampled_indices[idx_pos]
            
            self.gaussion_force_center[j][idx_force, 0] = flat_x_coords[flat_idx]
            self.gaussion_force_center[j][idx_force, 1] = flat_y_coords[flat_idx]
            
            strength_scale = 1.0 #min(5.0 * flat_prob[flat_idx] * 100, 1.0)
            
            # Random direction but with magnitude related to the adjoint field
            angle = np.random.uniform(0, 2 * np.pi)
            magnitude = np.random.uniform(0., 2e-2) * strength_scale
            
            self.gaussion_force_strength[j][idx_force, 0] = magnitude * np.cos(angle)
            self.gaussion_force_strength[j][idx_force, 1] = magnitude * np.sin(angle)
        
        print(f"Frame {j}: Repositioned {cutoff} forces based on adjoint passive scalar distribution")
        

    def init_lbfgsb(self, step_num):
        param_size = self.gaussion_force_num * 2 * 2  # control_num * (2 for center + 2 for strength)
        
        x = np.concatenate([
            self.gaussion_force_center[step_num].flatten(),  # [control_num, 2] -> [control_num*2]
            self.gaussion_force_strength[step_num].flatten()  # [control_num, 2] -> [control_num*2]
        ])
        
        self.lbfgsb_state = {
            'n': param_size,
            'x': x.copy(),
            'f': 0.0,
            'g': np.zeros(param_size),
            'H': np.eye(param_size),  # Initial Hessian approximation
            'nfev': 0,
            'njev': 0,
            'nit': 0,
            'status': 0,
            'message': '',
            'warnflag': 0,
            'task': b'START',
            'old_d': np.zeros(param_size),
            'rho': [],
            's': [],
            'y': [],
            'alpha': [],
            'sk': []
        }
        
        self.m = 10  # Number of corrections to approximate the inverse Hessian
        for i in range(self.m):
            self.lbfgsb_state['rho'].append(0.0)
            self.lbfgsb_state['s'].append(np.zeros(param_size))
            self.lbfgsb_state['y'].append(np.zeros(param_size))
            self.lbfgsb_state['alpha'].append(0.0)
            self.lbfgsb_state['sk'].append(0.0)


    def update_with_lbfgsb(self, step_num, adj_center, adj_strength, penalty_center, penalty_strength):
        old_center = self.gaussion_force_center[step_num].copy()
        old_strength = self.gaussion_force_strength[step_num].copy()
        
        x0 = np.concatenate([old_center.flatten(), old_strength.flatten()])
        
        center_size = old_center.size
        
        adj_center_flat = adj_center.flatten()
        penalty_center_flat = penalty_center.flatten()
        adj_strength_flat = adj_strength.flatten()
        penalty_strength_flat = penalty_strength.flatten()
        
        scaled_grad = np.zeros_like(x0)
        scaled_grad[:center_size] = self.alpha_center * adj_center_flat + self.alpha_norm_center * penalty_center_flat
        scaled_grad[center_size:] = self.alpha_strength * adj_strength_flat + self.alpha_norm_strength * penalty_strength_flat
        
        current_loss = self.simulator.loss if hasattr(self.simulator, 'loss') else 0.0
        
        trust_region_size = self.schedule_lr()
        
        def func(x):
            delta_x = x - x0
            
            linear_term = np.dot(scaled_grad, delta_x)
            
            quad_coef = 0.5 / trust_region_size
            quadratic_term = quad_coef * np.sum(delta_x**2)
            
            approx_loss = current_loss + linear_term + quadratic_term
            
            return approx_loss, scaled_grad
        
        lb_x, ub_x = 0.0, 1.0
        lb_y, ub_y = 0.0, 1.0
        
        bounds = []
        for i in range(self.gaussion_force_num):
            bounds.append((lb_x, ub_x))
            bounds.append((lb_y, ub_y))
        
        for i in range(self.gaussion_force_num * 2):
            bounds.append((None, None))
        
        new_x, f, d = fmin_l_bfgs_b(
            func, 
            x0, 
            bounds=bounds, 
            maxiter=1,
            maxfun=5, 
            m=10,     
            factr=1e7,
            pgtol=1e-3
        )
        
        new_center = new_x[:center_size].reshape(old_center.shape)
        new_strength = new_x[center_size:].reshape(old_strength.shape)
        
        self.gaussion_force_center[step_num] = new_center
        self.gaussion_force_strength[step_num] = new_strength
        
    def calculate_dtheta(self, j, adj_gaussion_force_strength, adj_gaussion_force_center, 
                 penalty_gaussion_force_strength, penalty_gaussion_force_center):
        def print_min_max(name, array):
            print(f"{name}: min={np.min(array):.6e}, max={np.max(array):.6e}")
        
        print_min_max("adj_gaussion_force_strength", adj_gaussion_force_strength)
        print_min_max("adj_gaussion_force_center", adj_gaussion_force_center)
        print_min_max("penalty_gaussion_force_strength", penalty_gaussion_force_strength)
        print_min_max("penalty_gaussion_force_center", penalty_gaussion_force_center)
        
        # Store values before update for comparison
        force_center_before = self.gaussion_force_center[j].copy()
        force_strength_before = self.gaussion_force_strength[j].copy()
        
        if self.use_lbfgsb:
            print("Using Newton!")
            self.update_with_lbfgsb(j, 
                                    adj_gaussion_force_center,
                                    adj_gaussion_force_strength,
                                    penalty_gaussion_force_center,
                                    penalty_gaussion_force_strength)
        else:
            self.gaussion_force_center[j] -= self.alpha_center * adj_gaussion_force_center
            self.gaussion_force_strength[j] -= self.alpha_strength * adj_gaussion_force_strength
            self.gaussion_force_center[j] -= self.alpha_norm_center * penalty_gaussion_force_center
            self.gaussion_force_strength[j] -= self.alpha_norm_strength * penalty_gaussion_force_strength
        
        def check_nan(array, name):
            if np.isnan(array).any():
                print(f"⚠ Updated {name} contains NaN values!")
            else:
                print(f"✅ Updated {name} has no NaN values.")
        
        check_nan(self.gaussion_force_center[j], "gaussion_force_center")
        check_nan(self.gaussion_force_strength[j], "gaussion_force_strength")
        
        def print_before_after(name, before, after):
            print(f"{name} BEFORE: min={np.min(before):.6e}, max={np.max(before):.6e}")
            print(f"{name} AFTER : min={np.min(after):.6e}, max={np.max(after):.6e}")
        
        print_before_after("gaussion_force_center", force_center_before, self.gaussion_force_center[j])
        print_before_after("gaussion_force_strength", force_strength_before, self.gaussion_force_strength[j])

    def plot_gaussian_forces(self, step_num, output_dir, frame_idx):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(10, 10))
        
        y, x = np.mgrid[0:1:256j, 0:1:256j]
        force_field_x = np.zeros_like(x)
        force_field_y = np.zeros_like(y)
        
        for i in range(self.gaussion_force_num):
            cx = self.gaussion_force_center[step_num][i, 0]
            cy = self.gaussion_force_center[step_num][i, 1]
            r = self.gaussion_force_radius[step_num][i]
            sx = self.gaussion_force_strength[step_num][i, 0]
            sy = self.gaussion_force_strength[step_num][i, 1]
            
            gaussian = np.exp(-r * ((x - cx)**2 + (y - cy)**2))
            force_field_x += sx * gaussian
            force_field_y += sy * gaussian
        
        magnitude = np.sqrt(force_field_x**2 + force_field_y**2)
        
        plt.streamplot(x[0, :], y[:, 0], force_field_x, force_field_y, 
                    density=1.5, color=magnitude, cmap='viridis',
                    linewidth=1.5, arrowsize=1.5)
        
        for i in range(self.gaussion_force_num):
            cx = self.gaussion_force_center[step_num][i, 0]
            cy = self.gaussion_force_center[step_num][i, 1]
            s = np.sqrt(self.gaussion_force_strength[step_num][i, 0]**2 + 
                    self.gaussion_force_strength[step_num][i, 1]**2)
            r = 1.0 / np.sqrt(self.gaussion_force_radius[step_num][i])
            
            # Scale the point size with radius and strength
            marker_size = max(r, 5) * 10
            color = 'red' if s > 0 else 'blue'
            alpha = min(abs(s) * 500, 1.0)  
            
            plt.scatter(cx, cy, s=marker_size, color=color, alpha=alpha, 
                    edgecolor='black', linewidth=1)
        
        # Customize the plot
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.title(f'Gaussian Force Field - Frame {frame_idx}')
        plt.colorbar(label='Force Magnitude')
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'gaussian_forces_{frame_idx:04d}.png'), dpi=dpi_vor)
        plt.close()

    def calc_target(self):
        logsdir = os.path.join('logs', exp_name+"_target")
        os.makedirs(logsdir, exist_ok=True)
        remove_everything_in(logsdir)
        vortdir = 'vorticity'
        vortdir = os.path.join(logsdir, vortdir)
        os.makedirs(vortdir, exist_ok=True)
        passivedir = 'passive'
        passivedir = os.path.join(logsdir, passivedir)
        os.makedirs(passivedir, exist_ok=True)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        os.makedirs(save_u_dir, exist_ok=True)
        target_final_dir = 'final'
        target_final_dir = os.path.join(save_u_dir , target_final_dir)     
        os.makedirs(target_final_dir, exist_ok=True)
        
        density = self.init_circle()
        total1 = np.sum(density)
        write_field(density, passivedir, 0, vmin=0, vmax=1, dpi=dpi_vor)
        file_path = os.path.join(target_final_dir, "s"+ f"_{0}.npy")
        np.save(file_path, density)

        
        density = self.init_dragon()
        total2 = np.sum(density)
        #density = density/total2*total1
        write_field(density, passivedir, total_steps, vmin=0, vmax=1, dpi=dpi_vor)
        file_path = os.path.join(target_final_dir, "s"+ f"_{total_steps}.npy")
        np.save(file_path, density)
    
    def init_circle(self, match_area_with_target=True):
        density = np.load("./data/density_G_arial_scaled_shifted_0.8.npy")
        return density
    
    

    def init_dragon(self):
        # Load the original density array (256x256)
        density = np.load("./data/density_R_arial_scaled_shifted_0.8.npy")
        return density

    def iter(self):
        self.opt_iter+=1 
        
            
        self.schedule_lr()
        logsdir = os.path.join('logs', exp_name)
        os.makedirs(logsdir, exist_ok=True)
        remove_everything_in(logsdir)
        vortdir = 'vorticity'
        vortdir = os.path.join(logsdir, vortdir)
        os.makedirs(vortdir, exist_ok=True)
        passivedir = 'passive'
        passivedir = os.path.join(logsdir, passivedir)
        os.makedirs(passivedir, exist_ok=True)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        os.makedirs(save_u_dir, exist_ok=True)
        gaussiandir = 'gaussian_forces'
        gaussiandir = os.path.join(logsdir, gaussiandir)
        os.makedirs(gaussiandir, exist_ok=True)
        shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')
        #self.simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir,self.save_u_target_dir)
        self.simulator.init(init_passive_np = self.init_circle())    
        w_numpy = self.simulator.calculate_w()
        w_max = 15
        w_min = -15
        write_field(w_numpy, vortdir, 0, vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
        if(add_passive_scalar):
            passive_numpy = self.simulator.passive.to_numpy()
            write_field(passive_numpy, passivedir, 0, vmin=0, vmax=1, dpi=dpi_vor)
        last_output_substep = 0
        # Forward Step
        while True:
            control_force_para={
                "r":self.gaussion_force_radius[self.simulator.step_num[None]+1],
                "c":self.gaussion_force_center[self.simulator.step_num[None]+1],
                "s":self.gaussion_force_strength[self.simulator.step_num[None]+1],
                "n":self.gaussion_force_num
            }
            output_frame, final_flag = self.simulator.forward_step_midpoint(True,True,control_force_para=control_force_para)
            if output_frame:
                w_numpy = self.simulator.calculate_w()     
                write_field(w_numpy, vortdir, self.simulator.frame_idx[None], vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    passive_numpy = self.simulator.passive.to_numpy()
                    write_field(passive_numpy, passivedir, self.simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                
                print("[Simulate] Finished frame: ", self.simulator.frame_idx[None], " in ", self.simulator.step_num[None]-last_output_substep, "substeps \n\n")
                last_output_substep = self.simulator.step_num[None]
            if final_flag:
                self.plot_gaussian_forces(self.simulator.step_num[None], gaussiandir, self.simulator.frame_idx[None])
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
        gpassivedir = 'gradient_passive'
        gpassivedir = os.path.join(logsdir, gpassivedir)
        os.makedirs(gpassivedir, exist_ok=True)
        w_max = 1.5#15*3
        w_min = -1.5#-15*3
        
        # Backward Step
        while True:
            control_force_para={
                "r":self.gaussion_force_radius[self.simulator.step_num[None]-1],
                "c":self.gaussion_force_center[self.simulator.step_num[None]-1],
                "s":self.gaussion_force_strength[self.simulator.step_num[None]-1],
                "n":self.gaussion_force_num
            }
            output_frame, final_flag = self.simulator.backtrack_step_midpoint(False,False,control_force_para=control_force_para)
            self.calculate_dtheta(self.simulator.step_num[None], 
                                  self.simulator.adj_gaussion_force_strength.to_numpy(),self.simulator.adj_gaussion_force_center.to_numpy(),
                                  self.simulator.penalty_gaussion_force_strength.to_numpy(),self.simulator.penalty_gaussion_force_center.to_numpy()
                                )
            if output_frame:
                w_numpy = self.simulator.calculate_gw()     
                write_field(w_numpy, gvortdir, self.simulator.frame_idx[None], vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    write_field(self.simulator.adj_passive.to_numpy(),gpassivedir, self.simulator.frame_idx[None], vmin=0,
                        vmax=1, dpi=dpi_vor)
                    
            
            if self.opt_iter % 30 == 0:
                self.reposition_weak_forces(self.simulator.step_num[None])

            if final_flag:
                self.plot_gaussian_forces(self.simulator.step_num[None], gaussiandir, self.simulator.frame_idx[None])
                break

        self.history_loss.append(self.simulator.loss)

def write_file(l,file):
    with open(file, "a") as f:
        for item in l:
            f.write(str(item) + ",")
        f.write("\n")

if __name__ == '__main__':
    opt = OptimizerSimple()
    simulate_target = True
    # if(simulate_target):
    opt.calc_target()
    # else:         
    log_file = "./shape_dragon.txt" 
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Deleted: {log_file}")
    else:
        print(f"File does not exist: {log_file}")
    write_file([f"iter:{opt.opt_iter}-"+"[loss:]"]+opt.history_loss,log_file)
    while(True):
        opt.iter()
        write_file([f"iter:{opt.opt_iter}-"+"[loss:]"]+opt.history_loss,log_file)
        center_np   = np.stack(opt.gaussion_force_center, axis=0)
        radius_np   = np.stack(opt.gaussion_force_radius, axis=0)
        strength_np = np.stack(opt.gaussion_force_strength, axis=0)

        print("center:", center_np.shape)     # (total_steps, control_num, 2)
        print("radius:", radius_np.shape)     # (total_steps, control_num)
        print("strength:", strength_np.shape) # (total_steps, control_num, 2)
        np.savez("checkpoint_gaussian_forces.npz",
                center=center_np,
                radius=radius_np,
                strength=strength_np)
        #data = np.load("checkpoint_gaussian_forces.npz")
        #center = data["center"]
        #radius = data["radius"]
        #strength = data["strength"]

