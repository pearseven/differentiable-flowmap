# 
from hyperparameters import *
from taichi_utils import *
from mgpcg_solid import *
#from mgpcg import *
from init_conditions import *
from io_utils import *
from flowmap import *
import sys,re,time


#dx = 1./res_y
ti.init(arch=ti.cuda, device_memory_GB=8.0, debug = False,default_fp = ti.f32)

class Timer:
    def __init__(self):
        self.start_times = {}
        self.elapsed_times = {}

    def start(self, tag):
        if(print_time):
            ti.sync()
            self.start_times[tag] = time.time()

    def end(self, tag):
        if(print_time):
            ti.sync()
            if tag not in self.start_times:
                print(f"[Timer Warning] No start() call found for tag '{tag}'")
                return
            elapsed = time.time() - self.start_times[tag]
            self.elapsed_times[tag] = elapsed
            print(f"[Timer] {tag} took {elapsed:.6f} seconds.")
            del self.start_times[tag]  # 可选：不保留 start 时间

    def print_all(self):
        if(print_time):
            print("=== Timer Summary ===")
            for tag, elapsed in self.elapsed_times.items():
                print(f"{tag}: {elapsed:.6f} seconds")
    def clear_all(self):
        self.start_times = {}
        self.elapsed_times = {}


class RW_Cache:
    def __init__(self):
        self.n_steps = 100
        cache_u = np.zeros((self.n_steps, res_x + 1, res_y+1, 2), dtype=np.float32)
        cache_adj_u = np.zeros((self.n_steps, res_x + 1, res_y+1, 2), dtype=np.float32)
        cache_s1 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_s2 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_s3 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_adj_s1 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_adj_s2 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_adj_s3 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_target_s1 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_target_s2 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_target_s3 = np.zeros((self.n_steps, res_x, res_y, 1), dtype=np.float32)
        cache_phi = np.zeros((self.n_steps,res_x, res_y, 2), dtype=np.float32)
        cache_phi_xy = np.zeros((self.n_steps,res_x+1, res_y+1, 4), dtype=np.float32)
        cache_F = np.zeros((self.n_steps,res_x+1, res_y+1,4), dtype=np.float32)
        #cache_psi = np.zeros((self.n_steps,res_x, res_y, 2), dtype=np.float32)
        #cache_psi_xy = np.zeros((self.n_steps,res_x+1, res_y+1, 4), dtype=np.float32)
        #cache_T = np.zeros((self.n_steps,res_x+1, res_y+1,4), dtype=np.float32)
        self.buffer={
            "u":{"buffer":cache_u,"info":[],"file_info":[]},
            "adj_u":{"buffer":cache_adj_u,"info":[],"file_info":[]},
            "s1":{"buffer":cache_s1,"info":[],"file_info":[]},
            "adj_s1":{"buffer":cache_adj_s1,"info":[],"file_info":[]},
            "target_s1":{"buffer":cache_target_s1,"info":[],"file_info":[]},
            "s2":{"buffer":cache_s2,"info":[],"file_info":[]},
            "adj_s2":{"buffer":cache_adj_s2,"info":[],"file_info":[]},
            "target_s2":{"buffer":cache_target_s2,"info":[],"file_info":[]},
            "s3":{"buffer":cache_s3,"info":[],"file_info":[]},
            "adj_s3":{"buffer":cache_adj_s3,"info":[],"file_info":[]},
            "target_s3":{"buffer":cache_target_s3,"info":[],"file_info":[]},
            "phi":{"buffer":cache_phi,"info":[],"file_info":[]},
            "phi_xy":{"buffer":cache_phi_xy,"info":[],"file_info":[]},
            "F":{"buffer":cache_F,"info":[],"file_info":[]},
            #"psi":{"buffer":cache_psi,"info":[],"file_info":[]},
            #"psi_xy":{"buffer":cache_psi_xy,"info":[],"file_info":[]},
            #"T":{"buffer":cache_T,"info":[],"file_info":[]},
        }
    def init(self):
        for name, iterms in self.buffer.items():
            self.buffer[name]["info"].clear()
            self.buffer[name]["file_info"].clear()


    def force_write(self,name, dir):
        buffer, info = self.buffer[name]["buffer"],self.buffer[name]["info"]
        if(len(info)>0):
            file_path = os.path.join(dir, name+ f"_{info[0]}_{info[-1]}.npy")
            np.save(file_path, buffer)
            info.clear()

    def write_disk_with_cache(self, data, name, dir, j):
        buffer, info = self.buffer[name]["buffer"],self.buffer[name]["info"]
        if(len(info)>=self.n_steps):
            file_path = os.path.join(dir, name+ f"_{info[0]}_{info[-1]}.npy")
            np.save(file_path, buffer)
            info.clear()
        data_np = data.to_numpy()
        if len(data_np.shape) == 4:
            s0, s1, s2, s3 = data_np.shape
            data_np = data_np.reshape(s0, s1, s2 * s3)
        elif len(data_np.shape) == 2:
            s0, s1= data_np.shape
            data_np = data_np.reshape(s0, s1, 1)
        ii, jj = data_np.shape[:2]
        buffer[len(info), 0:ii, 0:jj, :] = data_np
        info.append(int(j))

    def field_type(self,f):
        if hasattr(f, 'n') and hasattr(f, 'm'):
            if(f.m == 1):
                return 'v'
            else:
                return 'm'
        else:
                return 's'

    def from_np_to_field(self,np_array,data_field):
        ii,jj = data_field.shape
        if(self.field_type(data_field) == 's'):
            arr = np.squeeze(np_array)
            data_field.from_numpy(arr[:ii,:jj])
        elif(self.field_type(data_field) == 'v'):
            data_field.from_numpy(np_array[:ii,:jj,:])
        elif(self.field_type(data_field) == 'm'):
            arr = np_array.reshape(np_array.shape[0], np_array.shape[1], 2, 2)
            data_field.from_numpy(arr[:ii,:jj,:,:])

    def read_disk_with_cache(self, data_field, name, dir, j):
        buffer, info, file_info = self.buffer[name]["buffer"], self.buffer[name]["info"], self.buffer[name]["file_info"]
        if len(file_info) == 0:
            filenames = os.listdir(dir)
            pattern = re.compile(r"^.*?_(\d+)_(\d+)\.\w+$")
            for fname in filenames:
                match = pattern.match(fname)
                if match:
                    num1, num2 = int(match.group(1)), int(match.group(2))
                    file_info.append((num1, num2, fname))
            file_info.sort(key=lambda x: x[0])
        
        if len(info) > 0 and info[0] <= j <= info[-1]:
            index = info.index(j)
            self.from_np_to_field(buffer[index],data_field)
            return
        
        for num1, num2, fname in file_info:
            if num1 <= j <= num2:
                filepath = os.path.join(dir, fname)
                print(name,filepath)
                #print(self.buffer[name])
                data = np.load(filepath)                
                buffer[:num2 - num1 + 1] = data[:num2 - num1 + 1]
                info.clear()
                info.extend(range(num1, num2 + 1))
                index = info.index(j)
                self.from_np_to_field(buffer[index],data_field)
                return 

        raise ValueError(f"Frame {j} not found in any available file.")

    def write_disk(self,data,name, dir, j):
        if isinstance(data, float) or isinstance(data, int):
            np_data = np.array([data])
        elif isinstance(data, ti.Field) or isinstance(data, ti.MatrixField):
            np_data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            np_data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        file_path = os.path.join(dir, name+ f"_{j}.npy")
        np.save(file_path, np_data)

    def read_disk(self,name, dir, j):
        file_path = os.path.join(dir, name + f"_{j}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        np_data = np.load(file_path)
        return np_data

@ti.data_oriented
class LFM_Simulator:
    def __init__(self, res_x, res_y, dx,dt, reinit_every, log_dir, log_target_dir = None):
        self.dx = dx
        self.res_x = res_x
        self.res_y = res_y
        self.dt = dt
        self.reinit_every = reinit_every
        self.log_dir= log_dir
        self.log_target_dir = log_target_dir
        
        # Poisson solver
        self.boundary_mask = ti.field(float, shape=(self.res_x, self.res_y))
        self.boundary_vel = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.boundary_mask.fill(0.0)
        self.boundary_types = ti.Matrix([[2, 2], [2, 2]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
        self.solver = MGPCG_2_solid(self.boundary_types, self.boundary_mask, self.boundary_vel, N = [res_x, res_y])
        #self.solver = MGPCG_2(self.boundary_types, N = [self.res_x, self.res_y])

        self.X = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.X_x = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y))
        self.X_y = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1))
        center_coords_func(self.X, self.dx)
        horizontal_coords_func(self.X_x, self.dx)
        vertical_coords_func(self.X_y, self.dx)

        self.T_x = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # d_psi / d_x
        self.T_y = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # d_psi / d_y
        self.psi = ti.Vector.field(2, float, shape=(self.res_x, self.res_y)) # x coordinate
        self.psi_x = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # x coordinate
        self.psi_y = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # y coordinate
        self.F_x = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # d_phi / d_x
        self.F_y = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # d_phi / d_y
        self.phi = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.phi_x = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y))
        self.phi_y = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1))

        self.T_x_tem = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # d_psi / d_x
        self.T_y_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # d_psi / d_y
        self.psi_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y)) # x coordinate
        self.psi_x_tem = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # x coordinate
        self.psi_y_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # y coordinate
        self.F_x_tem = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y)) # d_phi / d_x
        self.F_y_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1)) # d_phi / d_y
        self.phi_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.phi_x_tem = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y))
        self.phi_y_tem = ti.Vector.field(2, float, shape=(self.res_x, self.res_y+1))

        self.u = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.w = ti.field(float, shape=(self.res_x, self.res_y))
        self.u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.tem_p = ti.field(float, shape=(self.res_x , self.res_y))

        self.init_u_x = ti.field(float, shape=(self.res_x+1, self.res_y)) # stores the "m0"
        self.init_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.err_u_x = ti.field(float, shape=(self.res_x+1, self.res_y)) # stores the roundtrip "m0"
        self.err_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.tmp_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.tmp_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        self.ad_u_x = ti.field(float, shape=(res_x + 1, res_y))
        self.ad_u_y = ti.field(float, shape=(res_x, res_y + 1))
        self.final_u_x = ti.field(float, shape=(res_x + 1, res_y))
        self.final_u_y = ti.field(float, shape=(res_x, res_y + 1))

        self.max_speed = ti.field(float, shape=())
        self.dts = ti.field(float, shape=(self.reinit_every))
        self.velocity_storage = {}
        
        self.step_num = ti.field(int, shape=())
        self.frame_idx = ti.field(int, shape=())
        self.step_num[None] = -1
        self.frame_idx[None] = 0

        self.disk_manage = RW_Cache()
        self.F_buffer = ti.Matrix.field(2,2, float, shape=(self.res_x+1, self.res_y+1))
        self.u_buffer = ti.Vector.field(2, float, shape=(self.res_x+1, self.res_y+1))
        self.phi_xy_buffer = ti.Matrix.field(2,2, float, shape=(self.res_x+1, self.res_y+1))

        self.vis_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.vis_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        self.timer = Timer()

        if(add_control_force):
            self.gaussion_force_num = control_num
            self.gaussion_force_real_num = ti.field(float,shape = ())
            self.gaussion_force_center = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.gaussion_force_inverse_radius = ti.field(float, shape=(self.gaussion_force_num))
            self.gaussion_force_strenth = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.control_force_x = ti.field(float, shape=(self.res_x+1, self.res_y))
            self.control_force_y = ti.field(float, shape=(self.res_x, self.res_y+1))


        if(add_passive_scalar):
            # passive only support for our method
            self.passive1 = ti.field(float, shape=(res_x, res_y))
            self.final_passive1 = ti.field(float, shape=(res_x, res_y))
            self.init_passive1 = ti.field(float, shape=(self.res_x, self.res_y))
            self.err_passive1 = ti.field(float, shape=(self.res_x, self.res_y))
            self.tmp_passive1 = ti.field(float, shape=(self.res_x, self.res_y))
            
            self.passive2 = ti.field(float, shape=(res_x, res_y))
            self.final_passive2 = ti.field(float, shape=(res_x, res_y))
            self.init_passive2 = ti.field(float, shape=(self.res_x, self.res_y))
            self.err_passive2 = ti.field(float, shape=(self.res_x, self.res_y))
            self.tmp_passive2 = ti.field(float, shape=(self.res_x, self.res_y))

            self.passive3 = ti.field(float, shape=(res_x, res_y))
            self.final_passive3 = ti.field(float, shape=(res_x, res_y))
            self.init_passive3 = ti.field(float, shape=(self.res_x, self.res_y))
            self.err_passive3 = ti.field(float, shape=(self.res_x, self.res_y))
            self.tmp_passive3 = ti.field(float, shape=(self.res_x, self.res_y))


    def apply_gausion_force(self, f_x, f_y, gaussion_force_center, gaussion_force_inverse_radius, gaussion_force_strenth, gaussion_num):
        self.gaussion_force_real_num[None] = gaussion_num
        self.gaussion_force_center.from_numpy(gaussion_force_center)
        self.gaussion_force_inverse_radius.from_numpy(gaussion_force_inverse_radius)
        self.gaussion_force_strenth.from_numpy(gaussion_force_strenth)
        self.apply_gausion_force_kernel(f_x,f_y)

    @ti.kernel
    def apply_gausion_force_kernel(self, f_x:ti.template(), f_y:ti.template()):
        f_x.fill(0.0)
        f_y.fill(0.0)
        for i,j in f_x:
            p = self.X_x[i,j]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    f_x[i,j]+= ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][0]
        for i,j in f_y:
            p = self.X_y[i,j]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    f_y[i,j]+= ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][1]
            

    @ti.kernel
    def calc_max_speed(self,u_x: ti.template(), u_y: ti.template()):
        self.max_speed[None] = 1.e-3 # avoid dividing by zero
        for i, j in ti.ndrange(self.res_x, self.res_y):
            u = 0.5 * (self.u_x[i, j] + self.u_x[i+1, j])
            v = 0.5 * (self.u_y[i, j] + self.u_y[i, j+1])
            speed = ti.sqrt(u ** 2 + v ** 2)
            ti.atomic_max(self.max_speed[None], speed)

    def march_no_neural(self,psi_x, T_x, psi_y, T_y, psi, step):
        # query neural buffer
        self.tmp_u_x.from_numpy(self.velocity_storage[step]["u_x"])
        self.tmp_u_y.from_numpy(self.velocity_storage[step]["u_y"])
        # time integration
        RK_grid(psi_x, T_x, self.tmp_u_x, self.tmp_u_y, self.dx, self.dt)
        RK_grid(psi_y, T_y, self.tmp_u_x, self.tmp_u_y, self.dx, self.dt)
        RK_grid_only_psi(psi, self.tmp_u_x, self.tmp_u_y, self.dx,self.dt)

    def backtrack_psi_grid(self,curr_step):
        reset_to_identity(self.psi, self.psi_x, self.psi_y, self.T_x, self.T_y,self.X,self.X_x,self.X_y)
        RK_grid_only_psi(self.psi, self.u_x, self.u_y, self.dx,self.dt)
        RK_grid(self.psi_x, self.T_x, self.u_x, self.u_y,  self.dx,self.dt)
        RK_grid(self.psi_y, self.T_y, self.u_x, self.u_y, self.dx,self.dt)
        for step in reversed(range(curr_step)):
            self.march_no_neural(self.psi_x, self.T_x, self.psi_y, self.T_y, self.psi, step)

    def march_phi_grid(self,curr_step):
        RK_grid_only_psi( self.phi, self.u_x, self.u_y, self.dx,-1 * self.dt)
        RK_grid(self.phi_x, self.F_x, self.u_x, self.u_y,self.dx, -1 * self.dt)
        RK_grid(self.phi_y, self.F_y, self.u_x, self.u_y,self.dx, -1 * self.dt)

    def write_velocity(self,u_x,u_y,ind):
        u_x_np= u_x.to_numpy()
        u_y_np= u_y.to_numpy()
        self.velocity_storage[ind] = {"u_x": u_x_np, "u_y": u_y_np}

    def delete_velocity(self,all_ind):
        for ind in range(all_ind):
            if ind in self.velocity_storage:
                del self.velocity_storage[ind]

    def init(self, theta  = None, forward = True, target = False, init_passive_np = None):
        self.midpoint_dir = 'midpoint'
        self.midpoint_dir = os.path.join(self.log_dir, self.midpoint_dir)
        os.makedirs(self.midpoint_dir, exist_ok=True)
        self.final_dir = 'final'
        self.final_dir = os.path.join(self.log_dir, self.final_dir)
        os.makedirs(self.final_dir, exist_ok=True)
        self.flowmap_F_dir = 'flowmap_F'
        self.flowmap_F_dir = os.path.join(self.log_dir, self.flowmap_F_dir)
        os.makedirs(self.flowmap_F_dir, exist_ok=True)
        self.flowmap_phi_dir = 'flowmap_phi'
        self.flowmap_phi_dir = os.path.join(self.log_dir, self.flowmap_phi_dir)
        os.makedirs(self.flowmap_phi_dir, exist_ok=True)
        self.flowmap_phi_xy_dir = 'flowmap_phi_xy'
        self.flowmap_phi_xy_dir = os.path.join(self.log_dir, self.flowmap_phi_xy_dir)
        os.makedirs(self.flowmap_phi_xy_dir, exist_ok=True)

        #self.flowmap_T_dir = 'flowmap_T'
        #self.flowmap_T_dir = os.path.join(self.log_dir, self.flowmap_T_dir)
        #os.makedirs(self.flowmap_T_dir, exist_ok=True)
        #self.flowmap_psi_dir = 'flowmap_psi'
        #self.flowmap_psi_dir = os.path.join(self.log_dir, self.flowmap_psi_dir)
        #os.makedirs(self.flowmap_psi_dir, exist_ok=True)
        #self.flowmap_psi_xy_dir = 'flowmap_psi_xy'
        #self.flowmap_psi_xy_dir = os.path.join(self.log_dir, self.flowmap_psi_xy_dir)
        #os.makedirs(self.flowmap_psi_xy_dir, exist_ok=True)


        self.passive_dir = "passive"
        self.passive_dir = os.path.join(self.log_dir, self.passive_dir)
        os.makedirs(self.passive_dir, exist_ok=True)
        self.disk_manage.init()

        self.viscosity = 0.0
        if forward:
            if target:
                self.boundary_mask.from_numpy(np.load(r"data/leaf_mask_256_scaled.npz")["mask"].astype(np.float32))
                self.boundary_vel.fill(0.0)
                eight_vortex_vel_func(self.u,self.X)
                #eight_vortex_vel_func_with_coef2(self.u,self.X)
                split_central_vector(self.u,self.u_x,self.u_y)
                self.solver.Poisson(self.u_x, self.u_y)
                if(add_passive_scalar):
                    simple_passive(self.passive1, self.passive2, self.passive3, self.X)
                    copy_to(self.passive1, self.init_passive1)
                    copy_to(self.passive2, self.init_passive2)
                    copy_to(self.passive3, self.init_passive3)
            else:
                self.boundary_mask.from_numpy(np.load(r"data/leaf_mask_256_scaled.npz")["mask"].astype(np.float32))
                self.boundary_vel.fill(0.0)
                sixteen_vortex_vel_func_with_pos_coef(self.u,self.X,ti.Vector(theta))
                split_central_vector(self.u,self.u_x,self.u_y)
                self.solver.Poisson(self.u_x, self.u_y)
                if(add_passive_scalar):
                    simple_passive(self.passive1, self.passive2, self.passive3, self.X)
                    copy_to(self.passive1, self.init_passive1)
                    copy_to(self.passive2, self.init_passive2)
                    copy_to(self.passive3, self.init_passive3)  

            mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y)
            if(add_passive_scalar):
                mask_passive_by_boundary(self.boundary_mask,self.passive1)
                mask_passive_by_boundary(self.boundary_mask,self.passive2)
                mask_passive_by_boundary(self.boundary_mask,self.passive3)

            self.step_num[None] = -1
            self.frame_idx[None] = 0
            reset_to_identity(self.phi, self.phi_x, self.phi_y, self.F_x, self.F_y,self.X,self.X_x,self.X_y)
            copy_to(self.u_x, self.init_u_x)
            copy_to(self.u_y, self.init_u_y)

    def calculate_w(self):
        get_central_vector(self.u_x, self.u_y, self.u)
        curl(self.u, self.w, dx)
        w_numpy = self.w.to_numpy()
        return w_numpy

    def final_condition(self):
        if(self.step_num[None] == total_steps - 1):
            return True
        else:
            return False

    def read_disk(self,name, dir, j):
        file_path = os.path.join(dir, name + f"_{j}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        np_data = np.load(file_path)
        return np_data

    def write_disk(self,data,name, dir, j):
        if isinstance(data, float) or isinstance(data, int):
            np_data = np.array([data])
        elif isinstance(data, ti.Field) or isinstance(data, ti.MatrixField):
            np_data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            np_data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        file_path = os.path.join(dir, name+ f"_{j}.npy")
        np.save(file_path, np_data)

    def step_midpoint(
        self, 
        write_flow_map = False,
        write_passive = False,
        control_force_para = None
    ):
        self.step_num[None] += 1
        j = self.step_num[None] % self.reinit_every
        self.calc_max_speed(self.u_x, self.u_y)
        output_frame = False
        CFL_dt =  CFL * self.dx / self.max_speed[None]
        curr_dt = self.dt
        print("CFL_dt",CFL_dt, "current dt",curr_dt)        
        if(self.step_num[None]%frame_per_step == frame_per_step-1 and output_image_frame):
            self.frame_idx[None]+=1
            output_frame = True
        if(add_passive_scalar and write_passive):
            self.disk_manage.write_disk_with_cache(self.passive1,"s1", self.passive_dir, self.step_num[None])
            self.disk_manage.write_disk_with_cache(self.passive2,"s2", self.passive_dir, self.step_num[None])
            self.disk_manage.write_disk_with_cache(self.passive3,"s3", self.passive_dir, self.step_num[None])
        
        self.step_flow_map(j,curr_dt,write_flow_map,write_passive,control_force_para = control_force_para )

        if(self.final_condition() or (sub_optimize==True and (self.step_num[None]+1)%sub_steps==0)):
            self.write_disk(self.u_x,"u_x", self.final_dir, self.step_num[None]+1)
            self.write_disk(self.u_y,"u_y", self.final_dir, self.step_num[None]+1)
            self.write_disk(self.step_num[None]+1,"final_ind", self.final_dir, 0)
            if(add_passive_scalar and write_passive):
                self.write_disk(self.passive1,"s1", self.final_dir, self.step_num[None]+1)
                self.write_disk(self.passive2,"s2", self.final_dir, self.step_num[None]+1)
                self.write_disk(self.passive3,"s3", self.final_dir, self.step_num[None]+1)
                self.disk_manage.force_write("s1",self.passive_dir)
                self.disk_manage.force_write("s2",self.passive_dir)
                self.disk_manage.force_write("s3",self.passive_dir)

            self.disk_manage.force_write("u",self.midpoint_dir)
            self.disk_manage.force_write("phi_xy",self.flowmap_phi_xy_dir)
            self.disk_manage.force_write("phi",self.flowmap_phi_dir)
            self.disk_manage.force_write("F",self.flowmap_F_dir)
            #self.disk_manage.force_write("psi_xy",self.flowmap_psi_xy_dir)
            #self.disk_manage.force_write("psi",self.flowmap_psi_dir)
            #self.disk_manage.force_write("T",self.flowmap_T_dir)
        print(self.step_num[None], self.final_condition())
        return output_frame, self.final_condition()
        
    def step_flow_map(self,j,curr_dt,write_flow_map, write_passive, 
                      control_force_para = None):
        
        self.timer.start("total_time")
        self.timer.start("first_adv")
        
        copy_to(self.u_x, self.ad_u_x)
        copy_to(self.u_y, self.ad_u_y)
        advect_u_grid(self.ad_u_x, self.ad_u_y,self.ad_u_x, self.ad_u_y, self.u_x, self.u_y, self.dx, 0.5*curr_dt, self.X_x,self.X_y)
        
        if(add_control_force):
            gaussion_force_center,gaussion_force_radius, gaussion_force_strenth, gaussion_num = control_force_para["c"],control_force_para["r"],control_force_para["s"],control_force_para["n"]
            
            self.timer.start("control_f")
            
            self.apply_gausion_force(self.control_force_x, self.control_force_y, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)
            
            self.timer.end("control_f")
            
            add_fields(self.u_x,self.control_force_x,self.u_x,0.5*curr_dt)
            add_fields(self.u_y,self.control_force_y,self.u_y,0.5*curr_dt)
        
        self.timer.end("first_adv")
        self.timer.start("first_possion")

        self.solver.Poisson(self.u_x, self.u_y)
        mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y)
        if(add_passive_scalar):
            mask_passive_by_boundary(self.boundary_mask,self.passive1)
            mask_passive_by_boundary(self.boundary_mask,self.passive2)
            mask_passive_by_boundary(self.boundary_mask,self.passive3)
        
        self.timer.end("first_possion")
        self.timer.start("second_adv")

        calculate_viscous_force(self.u_x,self.u_y,self.vis_x,self.vis_y,self.viscosity,dx)

        self.write_velocity(self.u_x, self.u_y, j)
        self.march_phi_grid(j)

        scalar2vec(self.u_x,self.u_y,self.u_buffer)
        self.disk_manage.write_disk_with_cache(self.u_buffer,"u",self.midpoint_dir, self.step_num[None])

        if(use_short_BFECC):
            reset_to_identity(self.psi_tem, self.psi_x_tem, self.psi_y_tem, self.T_x_tem, self.T_y_tem,self.X,self.X_x,self.X_y)
            reset_to_identity(self.phi_tem, self.phi_x_tem, self.phi_y_tem, self.F_x_tem, self.F_y_tem,self.X,self.X_x,self.X_y)
            if(add_passive_scalar):  
                RK_grid_only_psi(self.psi_tem, self.u_x, self.u_y, self.dx,curr_dt)
                RK_grid_only_psi(self.phi_tem, self.u_x, self.u_y, self.dx,-curr_dt)              
                BFECC_scalar(self.passive1, self.final_passive1,  self.err_passive1, self.tmp_passive1,   self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive1,self.passive1)
                BFECC_scalar(self.passive2, self.final_passive2,  self.err_passive2, self.tmp_passive2,   self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive2,self.passive2)
                BFECC_scalar(self.passive3, self.final_passive3,  self.err_passive3, self.tmp_passive3,   self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive3,self.passive3)
            RK_grid(self.psi_x_tem, self.T_x_tem, self.u_x, self.u_y, self.dx,curr_dt)
            RK_grid(self.psi_y_tem, self.T_y_tem, self.u_x, self.u_y, self.dx,curr_dt)
            RK_grid(self.phi_x_tem, self.F_x_tem, self.u_x, self.u_y, self.dx,-curr_dt)
            RK_grid(self.phi_y_tem, self.F_y_tem, self.u_x, self.u_y, self.dx,-curr_dt)
            BFECC(
                self.ad_u_x, self.ad_u_y, self.u_x, self.u_y, 
                self.err_u_x, self.err_u_y, self.tmp_u_x, self.tmp_u_y, 
                self.T_x_tem, self.T_y_tem, self.psi_x_tem, self.psi_y_tem,
                self.F_x_tem, self.F_y_tem, self.phi_x_tem, self.phi_y_tem,
                dx, BFECC_clamp
            )

        else:
            if(add_passive_scalar):
                advect_scalar_grid(self.u_x, self.u_y, self.passive1, self.final_passive1,  self.dx, curr_dt, self.X)
                copy_to(self.final_passive1,self.passive1)
                advect_scalar_grid(self.u_x, self.u_y, self.passive2, self.final_passive2,  self.dx, curr_dt, self.X)
                copy_to(self.final_passive2,self.passive2)
                advect_scalar_grid(self.u_x, self.u_y, self.passive3, self.final_passive3,  self.dx, curr_dt, self.X)
                copy_to(self.final_passive3,self.passive3)

            advect_u_grid(self.u_x, self.u_y, self.ad_u_x, self.ad_u_y, self.final_u_x, self.final_u_y, self.dx, curr_dt, self.X_x,self.X_y)
            copy_to(self.final_u_x,self.u_x)
            copy_to(self.final_u_y,self.u_y)
        
        if(add_control_force):
            gaussion_force_center,gaussion_force_radius, gaussion_force_strenth, gaussion_num = control_force_para["c"],control_force_para["r"],control_force_para["s"],control_force_para["n"]
            self.apply_gausion_force(self.control_force_x, self.control_force_y, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)
            add_fields(self.u_x,self.control_force_x,self.u_x,curr_dt)
            add_fields(self.u_y,self.control_force_y,self.u_y,curr_dt)
            accumulate_init(self.control_force_x, self.control_force_y, self.init_u_x,self.init_u_y, self.F_x, self.F_y, self.phi_x, self.phi_y, dx, curr_dt)

        self.timer.end("second_adv")
        self.timer.start("second_possion")
        
        add_fields(self.u_x,self.vis_x,self.u_x,curr_dt)
        add_fields(self.u_y,self.vis_y,self.u_y,curr_dt)
        accumulate_init(self.vis_x,self.vis_y, self.init_u_x,self.init_u_y, self.F_x, self.F_y, self.phi_x, self.phi_y, dx, curr_dt)

        self.solver.Poisson(self.u_x, self.u_y)
        mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y)
        if(add_passive_scalar):
            mask_passive_by_boundary(self.boundary_mask,self.passive1)
            mask_passive_by_boundary(self.boundary_mask,self.passive2)
            mask_passive_by_boundary(self.boundary_mask,self.passive3)
        self.timer.end("second_possion")
        self.timer.end("total_time")
        self.timer.print_all()
        self.timer.clear_all()

        if j == self.reinit_every-1:
            self.backtrack_psi_grid(self.reinit_every-1)
            BFECC(
                self.init_u_x, self.init_u_y, self.u_x, self.u_y, 
                self.err_u_x, self.err_u_y, self.tmp_u_x, self.tmp_u_y, 
                self.T_x, self.T_y, self.psi_x, self.psi_y,
                self.F_x, self.F_y, self.phi_x, self.phi_y,
                dx, BFECC_clamp
            )
            self.solver.Poisson(self.u_x, self.u_y)
            mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y)
            if(add_passive_scalar):
                mask_passive_by_boundary(self.boundary_mask,self.passive1)
                mask_passive_by_boundary(self.boundary_mask,self.passive2)
                mask_passive_by_boundary(self.boundary_mask,self.passive3)

            if(add_passive_scalar):
                BFECC_scalar(self.init_passive1, self.passive1,  self.err_passive1, self.tmp_passive1,   self.psi, self.phi, dx, BFECC_clamp)
                copy_to(self.passive1,self.init_passive1)
                BFECC_scalar(self.init_passive2, self.passive2,  self.err_passive2, self.tmp_passive2,   self.psi, self.phi, dx, BFECC_clamp)
                copy_to(self.passive2,self.init_passive2)
                BFECC_scalar(self.init_passive3, self.passive3,  self.err_passive3, self.tmp_passive3,   self.psi, self.phi, dx, BFECC_clamp)
                copy_to(self.passive3,self.init_passive3)

            if(write_flow_map):
                vec2mat(self.phi_x,self.phi_y,self.phi_xy_buffer)
                vec2mat(self.F_x,self.F_y,self.F_buffer)
                self.disk_manage.write_disk_with_cache(self.phi_xy_buffer,"phi_xy",self.flowmap_phi_xy_dir,int(self.step_num[None]/self.reinit_every))
                self.disk_manage.write_disk_with_cache(self.phi,"phi",self.flowmap_phi_dir,int(self.step_num[None]/self.reinit_every))
                self.disk_manage.write_disk_with_cache(self.F_buffer,"F",self.flowmap_F_dir,int(self.step_num[None]/self.reinit_every))
                vec2mat(self.psi_x,self.psi_y,self.phi_xy_buffer)
                vec2mat(self.T_x,self.T_y,self.F_buffer)
                #self.disk_manage.write_disk_with_cache(self.phi_xy_buffer,"psi_xy",self.flowmap_psi_xy_dir,int(self.step_num[None]/self.reinit_every))
                #self.disk_manage.write_disk_with_cache(self.psi,"psi",self.flowmap_psi_dir,int(self.step_num[None]/self.reinit_every))
                #self.disk_manage.write_disk_with_cache(self.F_buffer,"T",self.flowmap_T_dir,int(self.step_num[None]/self.reinit_every))
                
            reset_to_identity(self.phi, self.phi_x, self.phi_y, self.F_x, self.F_y, self.X,self.X_x,self.X_y)
            copy_to(self.u_x, self.init_u_x)
            copy_to(self.u_y, self.init_u_y)
            self.delete_velocity(self.reinit_every)

if __name__ == '__main__':
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

    simulator = LFM_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir)
    simulator.init()    
    w_numpy = simulator.calculate_w()
    w_max = 15
    w_min = -15

    write_field(w_numpy, vortdir, 0, vmin=w_min,
                vmax=w_max, dpi=dpi_vor)
    if(add_passive_scalar):
        passive_numpy = simulator.passive1.to_numpy()
        write_field(passive_numpy, passive1dir, 0, vmin=0, vmax=1, dpi=dpi_vor)
        passive_numpy = simulator.passive2.to_numpy()
        write_field(passive_numpy, passive2dir, 0, vmin=0, vmax=1, dpi=dpi_vor)
        passive_numpy = simulator.passive3.to_numpy()
        write_field(passive_numpy, passive3dir, 0, vmin=0, vmax=1, dpi=dpi_vor)

    last_output_substep = 0
    while True:
        output_frame, final_flag = simulator.step_midpoint(True,True)
        if output_frame:
            w_numpy = simulator.calculate_w()     
            write_field(w_numpy, vortdir, simulator.frame_idx[None], vmin=w_min,
                vmax=w_max, dpi=dpi_vor)
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
