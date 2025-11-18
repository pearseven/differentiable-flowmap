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
from solid_model import *

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
                1/(dx)**2,
                1/(dx)**2,
                1/(dx)**2,
                1/(2*dx)**2,
                1/(3*dx)**2,
                1/(4*dx)**2,
                1/(5*dx)**2,
                1/(6*dx)**2,
                1/(7*dx)**2,
                1/(8*dx)**2,
                1/(9*dx)**2
            ]
            self.gaussion_force_center = [np.zeros((control_num, 3)) for  i in range(total_steps)]
            self.gaussion_force_radius = [np.zeros((control_num)) for  i in range(total_steps)]
            self.gaussion_force_strength = [np.zeros((control_num, 3)) for  i in range(total_steps)]

            lb_x, ub_x = 0.0,1.0
            lb_y, ub_y = 0.0,1.0
            lb_z, ub_z = 0.0,1.0

            print(self.gaussion_force_radius)
            for j in range(total_steps):
                for i in range(self.gaussion_force_num):
                    self.gaussion_force_radius[j][i] = random.choice(self.gaussion_force_radius_range)
                    self.gaussion_force_center[j][i,0] = random.uniform(lb_x, ub_x)
                    self.gaussion_force_center[j][i,1] = random.uniform(lb_y, ub_y)
                    self.gaussion_force_center[j][i,2] = random.uniform(lb_z, ub_z)

                    self.gaussion_force_strength[j][i,0] = 0.0
                    self.gaussion_force_strength[j][i,1] = 0.0
                    self.gaussion_force_strength[j][i,2] = 0.0

        else:
            self.gaussion_force_num = control_num
            self.gaussion_force_radius_range = [
                1/(dx)**2,
                1/(dx)**2,
                1/(dx)**2,
                1/(dx)**2,
                1/(2*dx)**2,
                1/(3*dx)**2,
                1/(4*dx)**2,
                1/(5*dx)**2,
                1/(6*dx)**2,
                1/(7*dx)**2,
                1/(8*dx)**2,
                1/(9*dx)**2
            ]
            data = np.load("checkpoint_gaussian_forces-Copy6.npz")
            center = data["center"]
            radius = data["radius"]
            strength = data["strength"]
            
            #self.gaussion_force_center = [np.zeros((control_num, 3)) for  i in range(total_steps)]
            #self.gaussion_force_radius = [np.zeros((control_num)) for  i in range(total_steps)]
            #self.gaussion_force_strength = [np.zeros((control_num, 3)) for  i in range(total_steps)]

            self.gaussion_force_center = [center[i,:control_num,:] for  i in range(total_steps)]
            self.gaussion_force_radius = [radius[i,:control_num] for  i in range(total_steps)]
            self.gaussion_force_strength = [strength[i,:control_num,:] for  i in range(total_steps)]
            
            """
            lb_x, ub_x = 0.0,1.0
            lb_y, ub_y = 0.0,1.0
            lb_z, ub_z = 0.0,1.0
            for j in range(total_steps):
                self.gaussion_force_center[j][0:4000] = center[j,:,:]
                self.gaussion_force_radius[j][0:4000] = radius[j,:]
                self.gaussion_force_strength[j][0:4000] = strength[j,:,:]
                for i in range(4000, self.gaussion_force_num):
                    self.gaussion_force_radius[j][i] = random.choice(self.gaussion_force_radius_range)
                    self.gaussion_force_center[j][i,0] = random.uniform(lb_x, ub_x)
                    self.gaussion_force_center[j][i,1] = random.uniform(lb_y, ub_y)
                    self.gaussion_force_center[j][i,2] = random.uniform(lb_z, ub_z)

                    self.gaussion_force_strength[j][i,0] = 0.0
                    self.gaussion_force_strength[j][i,1] = 0.0
                    self.gaussion_force_strength[j][i,2] = 0.0"""

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
        self.simulator = LFM_Diff_Simulator(res_x,res_y,res_z,dx, act_dt, reinit_every,save_u_dir,save_u_target_dir)
    
    def schedule_lr(self):
        self.alpha_center = 1e-5 /2 /2 #/2
        self.alpha_strength = 1e-6 *2 #/2 #/2 #/2  #/2 #/2 #/2 #/2 #/2
        self.alpha_norm_center = 0#1e-5 *4
        self.alpha_norm_strength = 0#1e-5*4


    def calculate_dtheta(self, j, adj_gaussion_force_strength,adj_gaussion_force_center, penalty_gaussion_force_strength,penalty_gaussion_force_center):
        def print_min_max(name, array):
            print(f"{name}: min={np.min(array):.6e}, max={np.max(array):.6e}")
        
        print_min_max("adj_gaussion_force_strength", adj_gaussion_force_strength)
        print_min_max("adj_gaussion_force_center", adj_gaussion_force_center)
        print_min_max("penalty_gaussion_force_strength", penalty_gaussion_force_strength)
        print_min_max("penalty_gaussion_force_center", penalty_gaussion_force_center)
        
        # Store values before update for comparison
        force_center_before = self.gaussion_force_center[j].copy()
        force_strength_before = self.gaussion_force_strength[j].copy()
        
        
        #self.gaussion_force_center[j] -= self.alpha_center*adj_gaussion_force_center
        #self.gaussion_force_strength[j] -= self.alpha_strength*adj_gaussion_force_strength
        #self.gaussion_force_center[j] -= self.alpha_norm_center * penalty_gaussion_force_center
        #self.gaussion_force_strength[j] -= self.alpha_norm_strength * penalty_gaussion_force_strength

        self.gaussion_force_center[j] -= self.alpha_center*adj_gaussion_force_center
        self.gaussion_force_strength[j] -= self.alpha_strength*adj_gaussion_force_strength
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
        
        density = self.init_sphere()        
        a = np.sum(density)
        w_numpy = self.simulator.calculate_w()
        w_norm = np.linalg.norm(w_numpy, axis = -1)        
        write_vtks(w_norm, density, vortdir, 0)

        file_path = os.path.join(target_final_dir, "s"+ f"_{0}.npy")
        np.save(file_path, density)

        density = self.init_bunny()
        b = np.sum(density)
        print((b/a)**(1.0/3))
        w_numpy = self.simulator.calculate_w()
        w_norm = np.linalg.norm(w_numpy, axis = -1)        
        write_vtks(w_norm, density, vortdir, total_steps)

        file_path = os.path.join(target_final_dir, "s"+ f"_{total_steps}.npy")
        np.save(file_path, density)
    
    def init_sphere(self):
        density = np.zeros((res_x, res_y,res_z), dtype=np.float32)
        density0 = np.load("./model/density_G_centered_1.2x_128x128.npy")
        for i in range(64-16,64+16):
            dist = 1-abs(i-64)/16
            density[:,:,i] = density0*dist**0.1
        return density
    
    def init_bunny(self):
        density = np.zeros((res_x, res_y,res_z), dtype=np.float32)
        density0 = np.load("./model/density_R_centered_1.2x_128x128.npy")
        for i in range(64-16,64+16):
            dist = 1-abs(i-64)/16
            density[:,:,i] = density0*dist**0.1
        return density

    def reposition_weak_forces(self, j):
        # Determine cut percentage based on iteration number
        cut_percent = 0.2
        print(f"Iteration {self.opt_iter}: Repositioning weak forces... Cut percentage: {cut_percent}")
        
        adj_passive_np = self.simulator.adj_passive.to_numpy()
    
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
        
        z_coords, y_coords, x_coords = np.mgrid[0:1:res_z*1j, 0:1:res_y*1j, 0:1:res_x*1j]
        flat_x_coords = x_coords.flatten()
        flat_y_coords = y_coords.flatten()
        flat_z_coords = z_coords.flatten()

        sampled_indices = np.random.choice(
            range(len(flat_prob)), 
            size=len(weak_indices), 
            p=flat_prob,
            replace=True
        )

        for idx_pos, idx_force in enumerate(weak_indices):
            radius_idx = np.random.randint(len(self.gaussion_force_radius_range)//2)
            self.gaussion_force_radius[j][idx_force] = self.gaussion_force_radius_range[radius_idx]
            
            flat_idx = sampled_indices[idx_pos]

            self.gaussion_force_center[j][idx_force, 0] = flat_x_coords[flat_idx]
            self.gaussion_force_center[j][idx_force, 1] = flat_y_coords[flat_idx]
            self.gaussion_force_center[j][idx_force, 2] = flat_z_coords[flat_idx]

            self.gaussion_force_strength[j][idx_force, 0] = 0.0
            self.gaussion_force_strength[j][idx_force, 1] = 0.0
            self.gaussion_force_strength[j][idx_force, 2] = 0.0
        
        print(f"Frame {j}: Repositioned {cutoff} forces based on adjoint passive scalar distribution")


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
        shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')
        self.simulator.init(init_passive_np = self.init_sphere())    
        
        w_numpy = self.simulator.calculate_w()
        w_norm = np.linalg.norm(w_numpy, axis = -1)
        passive_numpy = self.simulator.passive.to_numpy()
        write_vtks(w_norm,passive_numpy, vortdir, 0)

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
                w_numpy = self.simulator.calculate_w()
                w_norm = np.linalg.norm(w_numpy, axis = -1)
                passive_numpy = self.simulator.passive.to_numpy()
                write_vtks(w_norm,passive_numpy, vortdir,  self.simulator.frame_idx[None])
                print("[Simulate] Finished frame: ", self.simulator.frame_idx[None], " in ", self.simulator.step_num[None]-last_output_substep, "substeps \n\n")
                last_output_substep = self.simulator.step_num[None]
            if final_flag:
                break
        
        logsdir = os.path.join('logs', exp_name)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        self.simulator.init_gradient()
        gvortdir = 'gradient_vorticity'
        gvortdir = os.path.join(logsdir, gvortdir)
        os.makedirs(gvortdir, exist_ok=True)
        gpassivedir = 'gradient_passive'
        gpassivedir = os.path.join(logsdir, gpassivedir)
        os.makedirs(gpassivedir, exist_ok=True)
        
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
            
            if self.opt_iter % 30 == 1 and self.opt_iter < 250:
                self.reposition_weak_forces(self.simulator.step_num[None])

            if output_frame:
                w_numpy = self.simulator.calculate_gw()
                w_norm = np.linalg.norm(w_numpy, axis = -1)
                passive_numpy = self.simulator.adj_passive.to_numpy()
                write_vtks(w_norm,passive_numpy, gvortdir, self.simulator.frame_idx[None])
            if final_flag:
                break

        self.history_loss.append(self.simulator.loss)

def write_file(l,file):
    with open(file, "a") as f:
        for item in l:
            f.write(str(item) + ",")
        f.write("\n")

if __name__ == '__main__':
    opt = OptimizerSimple()
    # simulate_target = False
    # if(simulate_target):
    opt.calc_target()
    # else:         
    log_file = "./G_to_R.txt" 
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

        print("center:", center_np.shape)    
        print("radius:", radius_np.shape)    
        print("strength:", strength_np.shape)
        np.savez("checkpoint_gaussian_forces.npz",
                center=center_np,
                radius=radius_np,
                strength=strength_np
                )

