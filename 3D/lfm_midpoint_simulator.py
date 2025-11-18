# 
from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
#from mgpcg import *
from fast_mgpcg import *
from init_conditions import *
from io_utils import *
from flowmap import *
import sys,re,time


ti.init(arch=ti.cuda, debug = False,default_fp = ti.f32)

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
        cache_u = np.zeros((self.n_steps, res_x + 1, res_y+1, res_z+1, 3), dtype=np.float32)
        cache_adj_u = np.zeros((self.n_steps, res_x + 1, res_y+1, res_z+1, 3), dtype=np.float32)
        cache_s = np.zeros((self.n_steps, res_x, res_y, res_z, 1), dtype=np.float32)
        cache_adj_s = np.zeros((self.n_steps, res_x, res_y,res_z, 1), dtype=np.float32)
        cache_target_s = np.zeros((self.n_steps, res_x, res_y,res_z, 1), dtype=np.float32)
        cache_phi = np.zeros((self.n_steps,res_x, res_y,res_z, 3), dtype=np.float32)
        cache_phi_xyz = np.zeros((self.n_steps,res_x+1, res_y+1,res_z+1, 9), dtype=np.float32)
        cache_F = np.zeros((self.n_steps,res_x+1, res_y+1, res_z+1,9), dtype=np.float32)
        self.buffer={
            "u":{"buffer":cache_u,"info":[],"file_info":[]},
            "adj_u":{"buffer":cache_adj_u,"info":[],"file_info":[]},
            "s":{"buffer":cache_s,"info":[],"file_info":[]},
            "adj_s":{"buffer":cache_adj_s,"info":[],"file_info":[]},
            "target_s":{"buffer":cache_target_s,"info":[],"file_info":[]},
            "phi":{"buffer":cache_phi,"info":[],"file_info":[]},
            "phi_xyz":{"buffer":cache_phi_xyz,"info":[],"file_info":[]},
            "F":{"buffer":cache_F,"info":[],"file_info":[]},
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
        if len(data_np.shape) == 5:
            s0, s1, s2, s3, s4 = data_np.shape
            data_np = data_np.reshape(s0, s1, s2, s3 * s4)
        elif len(data_np.shape) == 3:
            s0, s1, s2= data_np.shape
            data_np = data_np.reshape(s0, s1, s2, 1)
        ii, jj, kk = data_np.shape[:3]
        buffer[len(info), 0:ii, 0:jj, 0:kk, :] = data_np
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
        ii, jj, kk = data_field.shape
        if(self.field_type(data_field) == 's'):
            arr = np.squeeze(np_array)
            data_field.from_numpy(arr[:ii,:jj,:kk])
        elif(self.field_type(data_field) == 'v'):
            data_field.from_numpy(np_array[:ii,:jj,:kk, :])
        elif(self.field_type(data_field) == 'm'):
            arr = np_array.reshape(np_array.shape[0], np_array.shape[1],np_array.shape[2], 3, 3)
            data_field.from_numpy(arr[:ii,:jj,:kk, :,:])

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
    def __init__(self, res_x, res_y, res_z, dx, dt, reinit_every, log_dir, log_target_dir = None):
        self.dx = dx
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        self.dt = dt
        self.reinit_every = reinit_every
        self.log_dir= log_dir
        self.log_target_dir = log_target_dir
        
        # Poisson solver
        self.boundary_mask = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))
        self.boundary_vel = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.boundary_mask.fill(0.0)
        self.boundary_types = ti.Matrix([[2, 2], [2, 2], [2, 2]], ti.i32) # boundaries: 1 means Dirichlet, 2 means Neumann
        #self.solver = MGPCG_Solid_3(self.boundary_types, self.boundary_mask, self.boundary_vel, N = [res_x, res_y, res_z])
        self.solver = MGPCG_3(self.boundary_types, N = [self.res_x, self.res_y, self.res_z])

        self.X = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.X_x = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z))
        self.X_y = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.X_z = ti.Vector.field(3,float, shape=(self.res_x, self.res_y, self.res_z+1))
        center_coords_func(self.X, self.dx)
        x_coords_func(self.X_x, self.dx)
        y_coords_func(self.X_y, self.dx)
        z_coords_func(self.X_z, self.dx)

        self.T_x = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # d_psi / d_x
        self.T_y = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # d_psi / d_y
        self.T_z = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # d_psi / d_z

        self.psi = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z)) # x coordinate
        self.psi_x = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # x coordinate
        self.psi_y = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # y coordinate
        self.psi_z = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # y coordinate

        self.F_x = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # d_phi / d_x
        self.F_y = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # d_phi / d_y
        self.F_z = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # d_phi / d_z

        self.phi = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.phi_x = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z))
        self.phi_y = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.phi_z = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1))

        self.T_x_tem = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # d_psi / d_x
        self.T_y_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # d_psi / d_y
        self.T_z_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # d_psi / d_z

        self.psi_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z)) # x coordinate
        self.psi_x_tem = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # x coordinate
        self.psi_y_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # y coordinate
        self.psi_z_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # y coordinate
        
        self.F_x_tem = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z)) # d_phi / d_x
        self.F_y_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z)) # d_phi / d_y
        self.F_z_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1)) # d_phi / d_z

        self.phi_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.phi_x_tem = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y, self.res_z))
        self.phi_y_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.phi_z_tem = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z+1))

        self.u = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.w = ti.Vector.field(3, float, shape=(self.res_x, self.res_y, self.res_z))
        self.u_x = ti.field(float, shape=(self.res_x+1, self.res_y, self.res_z))
        self.u_y = ti.field(float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z+1))

        self.tem_p = ti.field(float, shape=(self.res_x , self.res_y, self.res_z))

        self.init_u_x = ti.field(float, shape=(self.res_x+1, self.res_y, self.res_z)) # stores the "m0"
        self.init_u_y = ti.field(float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.init_u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z+1))
        
        self.err_u_x = ti.field(float, shape=(self.res_x+1, self.res_y, self.res_z)) # stores the roundtrip "m0"
        self.err_u_y = ti.field(float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.err_u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z+1))

        self.tmp_u_x = ti.field(float, shape=(self.res_x+1, self.res_y, self.res_z))
        self.tmp_u_y = ti.field(float, shape=(self.res_x, self.res_y+1, self.res_z))
        self.tmp_u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z+1))

        self.ad_u_x = ti.field(float, shape=(self.res_x + 1, self.res_y, self.res_z))
        self.ad_u_y = ti.field(float, shape=(self.res_x, self.res_y + 1, self.res_z))
        self.ad_u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z + 1))

        self.final_u_x = ti.field(float, shape=(self.res_x + 1, self.res_y, self.res_z))
        self.final_u_y = ti.field(float, shape=(self.res_x, self.res_y + 1, self.res_z))
        self.final_u_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z + 1))

        self.max_speed = ti.field(float, shape=())
        self.dts = ti.field(float, shape=(self.reinit_every))
        self.velocity_storage = {}
        
        self.step_num = ti.field(int, shape=())
        self.frame_idx = ti.field(int, shape=())
        self.step_num[None] = -1
        self.frame_idx[None] = 0

        self.disk_manage = RW_Cache()
        self.F_buffer = ti.Matrix.field(3,3, float, shape=(self.res_x+1, self.res_y+1, self.res_z+1))
        self.u_buffer = ti.Vector.field(3, float, shape=(self.res_x+1, self.res_y+1, self.res_z+1))
        self.phi_xyz_buffer = ti.Matrix.field(3,3, float, shape=(self.res_x+1, self.res_y+1, self.res_z+1))

        self.timer = Timer()

        if(add_control_force):
            self.gaussion_force_num = control_num
            self.gaussion_force_real_num = ti.field(float,shape = ())
            self.gaussion_force_center = ti.Vector.field(3, float, shape=(self.gaussion_force_num))
            self.gaussion_force_inverse_radius = ti.field(float, shape=(self.gaussion_force_num))
            self.gaussion_force_strenth = ti.Vector.field(3, float, shape=(self.gaussion_force_num))
            self.control_force_x = ti.field(float, shape=(self.res_x+1, self.res_y, self.res_z))
            self.control_force_y = ti.field(float, shape=(self.res_x, self.res_y+1, self.res_z))
            self.control_force_z = ti.field(float, shape=(self.res_x, self.res_y, self.res_z+1))


        #if(add_passive_scalar):
        if(True):
            # passive only support for our method
            self.passive = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))
            self.final_passive = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))
            self.init_passive = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))
            self.err_passive = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))
            self.tmp_passive = ti.field(float, shape=(self.res_x, self.res_y, self.res_z))

        #self.segment_num = ti.field(int,shape=())
        #max_segment_num = 256*128*128#40*20*20
        #self.segment_x_p = ti.Vector.field(3, float,shape=(max_segment_num))
        #self.segment_x_n = ti.Vector.field(3, float,shape=(max_segment_num))
        #self.segment_strength = ti.field(float,shape=(max_segment_num))
        #self.segment_radius = ti.field(float,shape=(max_segment_num))

        #self.adj_segment_x_p = ti.Vector.field(3, float,shape=(max_segment_num))
        #self.adj_segment_x_n = ti.Vector.field(3, float,shape=(max_segment_num))
        #self.adj_segment_strength = ti.field(float,shape=(max_segment_num))
        #self.adj_segment_radius = ti.field(float,shape=(max_segment_num))    


    def apply_gausion_force(self, f_x, f_y, f_z, gaussion_force_center, gaussion_force_inverse_radius, gaussion_force_strenth, gaussion_num):
        self.gaussion_force_real_num[None] = gaussion_num
        self.gaussion_force_center.from_numpy(gaussion_force_center)
        self.gaussion_force_inverse_radius.from_numpy(gaussion_force_inverse_radius)
        self.gaussion_force_strenth.from_numpy(gaussion_force_strenth)
        self.apply_gausion_force_kernel(f_x,f_y, f_z)

    @ti.kernel
    def apply_gausion_force_kernel(self, f_x:ti.template(), f_y:ti.template(), f_z:ti.template()):
        f_x.fill(0.0)
        f_y.fill(0.0)
        f_z.fill(0.0)
        
        for i,j,k in f_x:
            p = self.X_x[i,j,k]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    f_x[i,j,k]+= ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][0]
 
        for i,j,k in f_y:
            p = self.X_y[i,j,k]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    f_y[i,j,k]+= ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][1]

        for i,j,k in f_z:
            p = self.X_z[i,j,k]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    f_z[i,j,k]+= ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][2]
            

    @ti.kernel
    def calc_max_speed(self,u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        self.max_speed[None] = 1.e-3 # avoid dividing by zero
        for i, j, k  in ti.ndrange(self.res_x, self.res_y, self.res_z):
            u = 0.5 * (u_x[i, j, k] + u_x[i+1, j, k])
            v = 0.5 * (u_y[i, j, k] + u_y[i, j+1, k])
            w = 0.5 * (u_z[i, j, k] + u_z[i, j, k+1])
            speed = ti.sqrt(u ** 2 + v ** 2 + w**2)
            ti.atomic_max(self.max_speed[None], speed)

    def march_no_neural(self,psi_x, T_x, psi_y, T_y, psi_z, T_z, psi, step):
        # query neural buffer
        self.tmp_u_x.from_numpy(self.velocity_storage[step]["u_x"])
        self.tmp_u_y.from_numpy(self.velocity_storage[step]["u_y"])
        self.tmp_u_z.from_numpy(self.velocity_storage[step]["u_z"])
        # time integration
        RK_grid(psi_x, T_x, self.tmp_u_x, self.tmp_u_y, self.tmp_u_z, self.dx, self.dt)
        RK_grid(psi_y, T_y, self.tmp_u_x, self.tmp_u_y, self.tmp_u_z, self.dx, self.dt)
        RK_grid(psi_z, T_z, self.tmp_u_x, self.tmp_u_y, self.tmp_u_z, self.dx, self.dt)
        RK_grid_only_psi(psi, self.tmp_u_x, self.tmp_u_y,self.tmp_u_z, self.dx,self.dt)

    def backtrack_psi_grid(self,curr_step):
        reset_to_identity(self.psi, self.psi_x, self.psi_y,self.psi_z, self.T_x, self.T_y,self.T_z,self.X,self.X_x,self.X_y,self.X_z)
        RK_grid_only_psi(self.psi, self.u_x, self.u_y,self.u_z, self.dx,self.dt)
        RK_grid(self.psi_x, self.T_x, self.u_x, self.u_y, self.u_z, self.dx,self.dt)
        RK_grid(self.psi_y, self.T_y, self.u_x, self.u_y, self.u_z, self.dx,self.dt)
        RK_grid(self.psi_z, self.T_z, self.u_x, self.u_y, self.u_z, self.dx,self.dt)
        for step in reversed(range(curr_step)):
            self.march_no_neural(self.psi_x, self.T_x, self.psi_y, self.T_y, self.psi_z, self.T_z, self.psi, step)

    def march_phi_grid(self,curr_step):
        RK_grid_only_psi( self.phi, self.u_x, self.u_y, self.u_z, self.dx,-1 * self.dt)
        RK_grid(self.phi_x, self.F_x, self.u_x, self.u_y, self.u_z, self.dx, -1 * self.dt)
        RK_grid(self.phi_y, self.F_y, self.u_x, self.u_y, self.u_z, self.dx, -1 * self.dt)
        RK_grid(self.phi_z, self.F_z, self.u_x, self.u_y, self.u_z, self.dx, -1 * self.dt)

    def write_velocity(self,u_x,u_y,u_z,ind):
        u_x_np= u_x.to_numpy()
        u_y_np= u_y.to_numpy()
        u_z_np= u_z.to_numpy()
        self.velocity_storage[ind] = {"u_x": u_x_np, "u_y": u_y_np, "u_z": u_z_np}

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
        self.flowmap_phi_xyz_dir = 'flowmap_phi_xyz'
        self.flowmap_phi_xyz_dir = os.path.join(self.log_dir, self.flowmap_phi_xyz_dir)
        os.makedirs(self.flowmap_phi_xyz_dir, exist_ok=True)
        self.passive_dir = "passive"
        self.passive_dir = os.path.join(self.log_dir, self.passive_dir)
        os.makedirs(self.passive_dir, exist_ok=True)
        self.disk_manage.init()

        if forward:
            if(case == 2):
                set_boundary_mask(self.boundary_mask,self.boundary_vel)
                init_one_vort(self.X, self.u, self.passive, self.tmp_passive)
                split_central_vector(self.u, self.u_x, self.u_y, self.u_z)
                self.solver.Poisson(self.u_x, self.u_y, self.u_z)

            elif(case == 3):
                set_boundary_mask(self.boundary_mask,self.boundary_vel)
                init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                self.solver.Poisson(self.u_x, self.u_y, self.u_z)

            elif(case == 4):
                set_boundary_mask(self.boundary_mask,self.boundary_vel)
                init_vorts_leapfrog(self.X, self.u, self.passive, self.tmp_passive)
                split_central_vector(self.u,self.u_x,self.u_y,self.u_z)
                self.solver.Poisson(self.u_x, self.u_y, self.u_z)

            elif(case == 5):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.u_x.from_numpy(theta[0])
                    self.u_y.from_numpy(theta[1])
                    self.u_z.from_numpy(theta[2])
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)

            elif(case == 6):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.segment_x_p.from_numpy(theta[0])
                    self.segment_x_n.from_numpy(theta[1])
                    self.segment_strength.from_numpy(theta[2])
                    self.segment_radius.from_numpy(theta[3])
                    self.segment_num[None] = theta[0].shape[0]
                    segment_velocity(
                        self.segment_num[None],
                        self.segment_radius,
                        self.segment_strength,
                        self.segment_x_p,
                        self.segment_x_n,
                        self.X,
                        self.u
                    )
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)         

            elif(case == 7):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    #init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                    if(add_passive_scalar):
                        init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                        curl(self.u, self.w, dx)
                        set_leapfrog_smoke(self.X,self.w,self.passive)
                        #set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive) 
                        
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.u_x.from_numpy(theta[0])
                    self.u_y.from_numpy(theta[1])
                    self.u_z.from_numpy(theta[2])
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)  
                    if(add_passive_scalar):
                        init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                        curl(self.u, self.w, dx)
                        set_leapfrog_smoke(self.X,self.w,self.passive)
                        #set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive)

            elif(case == 8):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                    #init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                    if(add_passive_scalar):
                        init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                        curl(self.u, self.w, dx)
                        set_leapfrog_smoke(self.X,self.w,self.passive)
                        #set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive) 
                        
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.segment_num[None] = theta[0]
                    self.segment_x_p.from_numpy(theta[1])
                    self.segment_x_n.from_numpy(theta[2])
                    self.segment_radius.from_numpy(theta[3])
                    segment_velocity_origin(
                        self.segment_num[None],
                        self.segment_x_p,
                        self.segment_x_n,
                        self.segment_radius,
                        self.X,
                        self.u
                    )
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)  
                    if(add_passive_scalar):
                        init_vorts_leapfrog_with_coef(self.X, self.u, self.passive, self.tmp_passive,0.5)
                        curl(self.u, self.w, dx)
                        set_leapfrog_smoke(self.X,self.w,self.passive)
                        #set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive)  
            
            elif(case == 9):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    #init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    init_vorts_plume(self.X, self.u, self.passive)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                    if(add_passive_scalar):
                        set_simple_smoke(self.X,self.passive)
                        #set_plume_smoke(self.X,self.w,self.passive)
                        copy_to(self.passive, self.init_passive) 
                        
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.u_x.from_numpy(theta[0])
                    self.u_y.from_numpy(theta[1])
                    self.u_z.from_numpy(theta[2])
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)  
                    if(add_passive_scalar):
                        set_simple_smoke(self.X,self.passive)
                        #set_plume_smoke(self.X,self.w,self.passive)
                        copy_to(self.passive, self.init_passive)

            elif(case == 10):
                if(target):
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    init_vorts_plume(self.X, self.u, self.passive)
                    #init_one_vort_with_coef(self.X, self.u, self.passive, self.tmp_passive, theta)
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                    if(add_passive_scalar):
                        #set_plume_smoke(self.X,self.w,self.passive)
                        set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive) 
                        
                else:
                    set_boundary_mask(self.boundary_mask,self.boundary_vel)
                    self.segment_num[None] = theta[0]
                    self.segment_x_p.from_numpy(theta[1])
                    self.segment_x_n.from_numpy(theta[2])
                    self.segment_radius.from_numpy(theta[3])
                    segment_velocity_origin(
                        self.segment_num[None],
                        self.segment_x_p,
                        self.segment_x_n,
                        self.segment_radius,
                        self.X,
                        self.u
                    )
                    split_central_vector(self.u,self.u_x, self.u_y, self.u_z)
                    self.solver.Poisson(self.u_x, self.u_y, self.u_z)  
                    if(add_passive_scalar):
                        #set_plume_smoke(self.X,self.w,self.passive)
                        set_simple_smoke(self.X,self.passive)
                        copy_to(self.passive, self.init_passive)  

            elif(case == 11):
                set_boundary_mask(self.boundary_mask,self.boundary_vel)
                self.u_x.fill(0.0)
                self.u_y.fill(0.0)
                self.u_z.fill(0.0)
                if(theta is not None):
                    self.u_x.from_numpy(theta["u_x"])
                    self.u_y.from_numpy(theta["u_y"])
                    self.u_z.from_numpy(theta["u_z"])
                self.solver.Poisson(self.u_x, self.u_y, self.u_z)
                if(add_passive_scalar):
                    self.passive.from_numpy(init_passive_np)
                    copy_to(self.passive, self.init_passive) 
            
            mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y ,self.u_z)
            if(add_passive_scalar):
                mask_passive_by_boundary(self.boundary_mask,self.passive)

            self.step_num[None] = -1
            self.frame_idx[None] = 0
            reset_to_identity(self.phi, self.phi_x, self.phi_y, self.phi_z, self.F_x, self.F_y, self.F_z, self.X,self.X_x,self.X_y ,self.X_z)
            copy_to(self.u_x, self.init_u_x)
            copy_to(self.u_y, self.init_u_y)
            copy_to(self.u_z, self.init_u_z)

    def calculate_w(self):
        get_central_vector(self.u_x, self.u_y, self.u_z, self.u)
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
        self.calc_max_speed(self.u_x, self.u_y, self.u_z)
        output_frame = False
        CFL_dt =  CFL * self.dx / self.max_speed[None]
        curr_dt = self.dt
        print("CFL_dt",CFL_dt, "current dt",curr_dt)        
        if(self.step_num[None]%frame_per_step == frame_per_step-1 and output_image_frame):
            self.frame_idx[None]+=1
            output_frame = True
        if(add_passive_scalar and write_passive):
            self.disk_manage.write_disk_with_cache(self.passive,"s", self.passive_dir, self.step_num[None])
        
        self.step_flow_map(j,curr_dt,write_flow_map,write_passive,control_force_para = control_force_para )

        if(self.final_condition()):
            self.write_disk(self.u_x,"u_x", self.final_dir, self.step_num[None]+1)
            self.write_disk(self.u_y,"u_y", self.final_dir, self.step_num[None]+1)
            self.write_disk(self.u_z,"u_z", self.final_dir, self.step_num[None]+1)
            self.write_disk(self.step_num[None]+1,"final_ind", self.final_dir, 0)
            if(add_passive_scalar and write_passive):
                self.write_disk(self.passive,"s", self.final_dir, self.step_num[None]+1)
                self.disk_manage.force_write("s",self.passive_dir)
            self.disk_manage.force_write("u",self.midpoint_dir)
            self.disk_manage.force_write("phi_xyz",self.flowmap_phi_xyz_dir)
            self.disk_manage.force_write("phi",self.flowmap_phi_dir)
            self.disk_manage.force_write("F",self.flowmap_F_dir)
        print(self.step_num[None], self.final_condition())
        return output_frame, self.final_condition()
        
    def step_flow_map(self,j,curr_dt,write_flow_map, write_passive, 
                      control_force_para = None):
        
        self.timer.start("total_time")
        self.timer.start("first_adv")
        
        copy_to(self.u_x, self.ad_u_x)
        copy_to(self.u_y, self.ad_u_y)
        copy_to(self.u_z, self.ad_u_z)
        advect_u_grid(self.ad_u_x, self.ad_u_y, self.ad_u_z, self.ad_u_x, self.ad_u_y, self.ad_u_z, self.u_x, self.u_y, self.u_z, self.dx, 0.5*curr_dt, self.X_x, self.X_y, self.X_z)
        
        if(add_control_force):
            gaussion_force_center,gaussion_force_radius, gaussion_force_strenth, gaussion_num = control_force_para["c"],control_force_para["r"],control_force_para["s"],control_force_para["n"]
            
            self.timer.start("control_f")
            
            self.apply_gausion_force(self.control_force_x, self.control_force_y, self.control_force_z, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)
            
            self.timer.end("control_f")
            
            add_fields(self.u_x,self.control_force_x,self.u_x,0.5*curr_dt)
            add_fields(self.u_y,self.control_force_y,self.u_y,0.5*curr_dt)
            add_fields(self.u_z,self.control_force_z,self.u_z,0.5*curr_dt)
        
        self.timer.end("first_adv")
        self.timer.start("first_possion")

        self.solver.Poisson(self.u_x, self.u_y, self.u_z)
        mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y,self.u_z)
        if(add_passive_scalar):
            mask_passive_by_boundary(self.boundary_mask,self.passive)
        
        self.timer.end("first_possion")
        self.timer.start("second_adv")

        self.write_velocity(self.u_x, self.u_y, self.u_z, j)
        self.march_phi_grid(j)

        scalar2vec(self.u_x,self.u_y,self.u_z,self.u_buffer)
        self.disk_manage.write_disk_with_cache(self.u_buffer,"u",self.midpoint_dir, self.step_num[None])

        if(use_short_BFECC):
            reset_to_identity(self.psi_tem, self.psi_x_tem, self.psi_y_tem, self.psi_z_tem, self.T_x_tem, self.T_y_tem, self.T_z_tem, self.X,self.X_x,self.X_y,self.X_z)
            reset_to_identity(self.phi_tem, self.phi_x_tem, self.phi_y_tem, self.phi_z_tem, self.F_x_tem, self.F_y_tem, self.F_z_tem, self.X,self.X_x,self.X_y,self.X_z)
            
            if(add_passive_scalar):  
                RK_grid_only_psi(self.psi_tem, self.u_x, self.u_y, self.u_z, self.dx,curr_dt)
                RK_grid_only_psi(self.phi_tem, self.u_x, self.u_y, self.u_z, self.dx,-curr_dt)              
                BFECC_scalar(self.passive, self.final_passive,  self.err_passive, self.tmp_passive,   self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive,self.passive)
            RK_grid(self.psi_x_tem, self.T_x_tem, self.u_x, self.u_y, self.u_z, self.dx,curr_dt)
            RK_grid(self.psi_y_tem, self.T_y_tem, self.u_x, self.u_y, self.u_z, self.dx,curr_dt)
            RK_grid(self.psi_z_tem, self.T_z_tem, self.u_x, self.u_y, self.u_z, self.dx,curr_dt)
            RK_grid(self.phi_x_tem, self.F_x_tem, self.u_x, self.u_y, self.u_z, self.dx,-curr_dt)
            RK_grid(self.phi_y_tem, self.F_y_tem, self.u_x, self.u_y, self.u_z, self.dx,-curr_dt)
            RK_grid(self.phi_z_tem, self.F_z_tem, self.u_x, self.u_y, self.u_z, self.dx,-curr_dt)
            BFECC(
                self.ad_u_x, self.ad_u_y, self.ad_u_z, self.u_x, self.u_y, self.u_z, 
                self.err_u_x, self.err_u_y, self.err_u_z, self.tmp_u_x, self.tmp_u_y, self.tmp_u_z, 
                self.T_x_tem, self.T_y_tem, self.T_z_tem, self.psi_x_tem, self.psi_y_tem, self.psi_z_tem,
                self.F_x_tem, self.F_y_tem, self.F_z_tem, self.phi_x_tem, self.phi_y_tem, self.phi_z_tem,
                dx, BFECC_clamp
            )

        else:
            if(add_passive_scalar):
                advect_scalar_grid(self.u_x, self.u_y, self.u_z, self.passive, self.final_passive,  self.dx, curr_dt, self.X)
                copy_to(self.final_passive,self.passive)
            advect_u_grid(self.u_x, self.u_y, self.u_z, self.ad_u_x, self.ad_u_y, self.ad_u_z, self.final_u_x, self.final_u_y, self.final_u_z, self.dx, curr_dt, self.X_x,self.X_y,self.X_z)
            copy_to(self.final_u_x,self.u_x)
            copy_to(self.final_u_y,self.u_y)
            copy_to(self.final_u_z,self.u_z)
        
        if(add_control_force):
            gaussion_force_center,gaussion_force_radius, gaussion_force_strenth, gaussion_num = control_force_para["c"],control_force_para["r"],control_force_para["s"],control_force_para["n"]
            self.apply_gausion_force(self.control_force_x, self.control_force_y,self.control_force_z, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)
            add_fields(self.u_x,self.control_force_x,self.u_x,curr_dt)
            add_fields(self.u_y,self.control_force_y,self.u_y,curr_dt)
            add_fields(self.u_z,self.control_force_z,self.u_z,curr_dt)
            accumulate_init(self.control_force_x, self.control_force_y, self.control_force_z, self.init_u_x, self.init_u_y, self.init_u_z, self.F_x, self.F_y, self.F_z, self.phi_x, self.phi_y, self.phi_z, dx, curr_dt)

        self.timer.end("second_adv")
        self.timer.start("second_possion")
        
        self.solver.Poisson(self.u_x, self.u_y, self.u_z)
        mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y,self.u_z)
        if(add_passive_scalar):
            mask_passive_by_boundary(self.boundary_mask,self.passive)
        self.timer.end("second_possion")
        self.timer.end("total_time")
        self.timer.print_all()
        self.timer.clear_all()

        if j == self.reinit_every-1:
            self.backtrack_psi_grid(self.reinit_every-1)
            BFECC(
                self.init_u_x, self.init_u_y, self.init_u_z, self.u_x, self.u_y, self.u_z, 
                self.err_u_x, self.err_u_y, self.err_u_z, self.tmp_u_x, self.tmp_u_y, self.tmp_u_z, 
                self.T_x, self.T_y, self.T_z, self.psi_x, self.psi_y, self.psi_z,
                self.F_x, self.F_y, self.F_z, self.phi_x, self.phi_y, self.phi_z,
                dx, BFECC_clamp
            )
            self.solver.Poisson(self.u_x, self.u_y, self.u_z)
            mask_velocity_by_boundary(self.boundary_mask,self.boundary_vel,self.u_x,self.u_y,self.u_z)
            if(add_passive_scalar):
                mask_passive_by_boundary(self.boundary_mask,self.passive)

            if(add_passive_scalar):
                BFECC_scalar(self.init_passive, self.passive,  self.err_passive, self.tmp_passive,   self.psi, self.phi, dx, BFECC_clamp)
                copy_to(self.passive,self.init_passive)
            if(write_flow_map):
                vec2mat(self.phi_x,self.phi_y,self.phi_z,self.phi_xyz_buffer)
                vec2mat(self.F_x,self.F_y,self.F_z,self.F_buffer)
                self.disk_manage.write_disk_with_cache(self.phi_xyz_buffer,"phi_xyz",self.flowmap_phi_xyz_dir,int(self.step_num[None]/self.reinit_every))
                self.disk_manage.write_disk_with_cache(self.phi,"phi",self.flowmap_phi_dir,int(self.step_num[None]/self.reinit_every))
                self.disk_manage.write_disk_with_cache(self.F_buffer,"F",self.flowmap_F_dir,int(self.step_num[None]/self.reinit_every))
                vec2mat(self.psi_x,self.psi_y,self.psi_z,self.phi_xyz_buffer)
                vec2mat(self.T_x,self.T_y,self.T_z,self.F_buffer)
                
            reset_to_identity(self.phi, self.phi_x, self.phi_y,self.phi_z, self.F_x, self.F_y,self.F_z, self.X,self.X_x,self.X_y,self.X_z)
            copy_to(self.u_x, self.init_u_x)
            copy_to(self.u_y, self.init_u_y)
            copy_to(self.u_z, self.init_u_z)
            self.delete_velocity(self.reinit_every)

if __name__ == '__main__':
    logsdir = os.path.join('logs', exp_name)
    os.makedirs(logsdir, exist_ok=True)
    remove_everything_in(logsdir)
    vortdir = 'vtk'
    vortdir = os.path.join(logsdir, vortdir)
    os.makedirs(vortdir, exist_ok=True)
    save_u_dir = "save_u"
    save_u_dir = os.path.join(logsdir, save_u_dir)
    os.makedirs(save_u_dir, exist_ok=True)
    shutil.copyfile('./hyperparameters.py', f'{logsdir}/hyperparameters.py')

    simulator = LFM_Simulator(res_x,res_y,res_z,dx, act_dt, reinit_every,save_u_dir)
    simulator.init()    
    w_numpy = simulator.calculate_w()
    w_max = 15
    w_min = -15
    if(add_passive_scalar):
        passive_numpy = simulator.passive.to_numpy()
        w_norm = np.linalg.norm(w_numpy, axis = -1)
        write_vtks(w_norm, passive_numpy, vortdir, 0)
    else:
        w_norm = np.linalg.norm(w_numpy, axis = -1)
        write_vtks_w_only(w_norm, vortdir, 0)
    
    last_output_substep = 0
    while True:
        output_frame, final_flag = simulator.step_midpoint(True,True)
        if output_frame:
            w_numpy = simulator.calculate_w()     
            if(add_passive_scalar):
                passive_numpy = simulator.passive.to_numpy()
                w_norm = np.linalg.norm(w_numpy, axis = -1)
                write_vtks(w_norm, passive_numpy, vortdir, simulator.frame_idx[None])
            else:
                w_norm = np.linalg.norm(w_numpy, axis = -1)
                write_vtks_w_only(w_norm, vortdir, simulator.frame_idx[None])
            print("[Simulate] Finished frame: ", simulator.frame_idx[None], " in ", simulator.step_num[None]-last_output_substep, "substeps \n\n")
            last_output_substep = simulator.step_num[None]
        if final_flag:
            break   
