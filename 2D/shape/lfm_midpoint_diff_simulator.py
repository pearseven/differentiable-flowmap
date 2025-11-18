# 
from hyperparameters import *
from taichi_utils import *
#from mgpcg_solid import *
from mgpcg import *
from init_conditions import *
from io_utils import *
from flowmap import *
from lfm_midpoint_simulator import *
import sys


#dx = 1./res_y
ti.init(arch=ti.cuda, device_memory_GB=8.0, debug = False, default_fp = ti.f32)

@ti.data_oriented
class LFM_Diff_Simulator(LFM_Simulator):
    def __init__(self, res_x, res_y, dx,dt, reinit_every, save_u_dir,save_u_target_dir = None):
        super().__init__(res_x, res_y, dx,dt, reinit_every, save_u_dir,save_u_target_dir)
        self.test_u2 = ti.field(float, shape=(self.res_x+1, self.res_y))

        self.adj_u = ti.Vector.field(2, float, shape=(self.res_x, self.res_y))
        self.adj_u_norm = ti.field(float, shape=(self.res_x, self.res_y))
        self.adj_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.init_adj_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.init_adj_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.init_adj_passive = ti.field(float, shape=(self.res_x, self.res_y))
        
        self.err_adj_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.err_adj_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.tmp_adj_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.tmp_adj_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        self.adj_u_x_tem = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y_tem = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.adj_u_x_tem2 = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y_tem2 = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.adj_u_x_AB = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y_AB = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.adj_u_x_AB_tem = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y_AB_tem = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.adj_u_x_AB_tem2 = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_u_y_AB_tem2 = ti.field(float, shape=(self.res_x, self.res_y+1))
        self.mid_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.mid_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        self.partial_u_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.partial_u_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        self.final_ind = None

        self.adj_passive = ti.field(float, shape=(self.res_x, self.res_y))
        self.target_passive = ti.field(float, shape=(self.res_x, self.res_y))
        self.adj_passive_f_x = ti.field(float, shape=(self.res_x+1, self.res_y))
        self.adj_passive_f_y = ti.field(float, shape=(self.res_x, self.res_y+1))

        if(add_control_force):
            self.gaussion_force_num = control_num
            self.gaussion_force_real_num = ti.field(float,shape = ())
            self.adj_gaussion_force_center = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.adj_gaussion_force_strength = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.penalty_gaussion_force_center = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.penalty_gaussion_force_strength = ti.Vector.field(2, float, shape=(self.gaussion_force_num))
            self.adj_control_force_x = ti.field(float, shape=(self.res_x+1, self.res_y))
            self.adj_control_force_y = ti.field(float, shape=(self.res_x, self.res_y+1))

    def calculate_adj_gausion_force(self, adj_u_x, adj_u_y, gaussion_force_center, gaussion_force_inverse_radius, gaussion_force_strenth, gaussion_num):
        self.gaussion_force_real_num[None] = gaussion_num
        self.gaussion_force_center.from_numpy(gaussion_force_center)
        self.gaussion_force_inverse_radius.from_numpy(gaussion_force_inverse_radius)
        self.gaussion_force_strenth.from_numpy(gaussion_force_strenth)
        self.calculate_adj_gausion_force_kernel(adj_u_x,adj_u_y)

    @ti.kernel
    def calculate_adj_gausion_force_kernel(self, adj_u_x:ti.template(), adj_u_y:ti.template()):
        self.adj_gaussion_force_center.fill(0.0)
        self.adj_gaussion_force_strength.fill(0.0)
        self.penalty_gaussion_force_strength.fill(0.0)
        self.penalty_gaussion_force_center.fill(0.0)
        for i,j in adj_u_x:
            p = self.X_x[i,j]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    self.adj_gaussion_force_strength[I][0] += ti.math.exp(-d**2*r)*adj_u_x[i,j]
                    self.adj_gaussion_force_center[I]+=self.gaussion_force_strenth[I][0]*ti.math.exp(-d**2*r)*2*(p-self.gaussion_force_center[I])*r*adj_u_x[i,j]
                    f = ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][0]
                    self.penalty_gaussion_force_strength[I][0] += 2*f* ti.math.exp(-d**2*r)
                    self.penalty_gaussion_force_center[I] +=  2*f* self.gaussion_force_strenth[I][0]*ti.math.exp(-d**2*r)*2*(p-self.gaussion_force_center[I])*r

        for i,j in adj_u_y:
            p = self.X_y[i,j]
            for I in range(self.gaussion_force_real_num[None]):
                if((p-self.gaussion_force_center[I]).norm()**2*self.gaussion_force_inverse_radius[I] < 5):
                    d = (p-self.gaussion_force_center[I]).norm()
                    r = self.gaussion_force_inverse_radius[I]
                    self.adj_gaussion_force_strength[I][1] += ti.math.exp(-d**2*r)*adj_u_y[i,j]
                    self.adj_gaussion_force_center[I]+=self.gaussion_force_strenth[I][1]*ti.math.exp(-d**2*r)*2*(p-self.gaussion_force_center[I])*r*adj_u_y[i,j]
                    f = ti.math.exp(-d**2*r)*self.gaussion_force_strenth[I][1]
                    self.penalty_gaussion_force_strength[I][1] += 2*f* ti.math.exp(-d**2*r)
                    self.penalty_gaussion_force_center[I] +=  2*f* self.gaussion_force_strenth[I][1]*ti.math.exp(-d**2*r)*2*(p-self.gaussion_force_center[I])*r

    def forward_step_midpoint(
        self, 
        write_flow_map = False,
        write_passive = False,
        control_force_para = None      
    ):
        return self.step_midpoint(write_flow_map,write_passive,control_force_para)

    def init_gradient(self,theta = None):
        self.init(forward=False)

        self.gradient_dir = 'gradient'
        self.gradient_dir = os.path.join(self.log_dir, self.gradient_dir)
        os.makedirs(self.gradient_dir, exist_ok=True)

        self.gradient_dir2 = 'gradient2'
        self.gradient_dir2 = os.path.join(self.log_dir, self.gradient_dir2)
        os.makedirs(self.gradient_dir2, exist_ok=True)

        self.gradient_passive_dir = 'gradient_passive'
        self.gradient_passive_dir = os.path.join(self.log_dir, self.gradient_passive_dir)
        os.makedirs(self.gradient_passive_dir, exist_ok=True)

        self.target_final_dir = 'final'
        self.target_final_dir = os.path.join(self.log_target_dir , self.target_final_dir)        
        
        self.target_passive_dir = "passive"
        self.target_passive_dir = os.path.join(self.log_target_dir, self.target_passive_dir)

        self.disk_manage.init()
        
        self.target_final_dir = 'final'
        self.target_final_dir = os.path.join(self.log_target_dir , self.target_final_dir)
        self.final_ind = int(self.read_disk("final_ind", self.final_dir, 0)[0])
        final_u_x_np = self.read_disk("u_x", self.final_dir, self.final_ind)
        final_u_y_np = self.read_disk("u_y", self.final_dir, self.final_ind)
        self.u_x.from_numpy(final_u_x_np)
        self.u_y.from_numpy(final_u_y_np)             
        self.adj_u_x.fill(0.0)
        self.adj_u_y.fill(0.0)
        if(add_passive_scalar):
            adj_passive_np = self.read_disk("s", self.final_dir, self.final_ind)
            target_passive_np = self.read_disk("s", self.target_final_dir, self.final_ind)
            self.adj_passive.from_numpy(2*(adj_passive_np-target_passive_np))
            self.passive.from_numpy(adj_passive_np)
            self.loss = np.sum((adj_passive_np-target_passive_np)**2)

        self.step_num[None] = self.final_ind
        copy_to(self.adj_u_x, self.init_adj_u_x)
        copy_to(self.adj_u_y, self.init_adj_u_y)
        if(add_passive_scalar):
            copy_to(self.adj_passive,self.init_adj_passive)
        if backward_u:
            copy_to(self.u_x, self.init_u_x)
            copy_to(self.u_y, self.init_u_y)
            if(add_passive_scalar):
                copy_to(self.passive,self.init_passive)
        reset_to_identity(self.psi, self.psi_x, self.psi_y, self.T_x, self.T_y,self.X,self.X_x,self.X_y)
        self.frame_idx[None] = int((self.step_num[None]-1)/frame_per_step)

    def calculate_gw(self):
        get_central_vector(self.adj_u_x, self.adj_u_y, self.u)
        curl(self.u, self.w, dx)
        w_numpy = self.w.to_numpy()
        return w_numpy
       
    def march_psi_grid(self,dt):
        RK_grid_only_psi( self.psi, self.mid_u_x, self.mid_u_y, self.dx, dt)
        RK_grid(self.psi_x, self.T_x, self.mid_u_x, self.mid_u_y,self.dx, dt)
        RK_grid(self.psi_y, self.T_y, self.mid_u_x, self.mid_u_y,self.dx, dt)

    def backtrack_step_midpoint(
        self,
        write_gradient = False,
        write_passive = False,
        control_force_para = None
    ):
        self.step_num[None] -= 1
        j = self.step_num[None] % self.reinit_every
        curr_dt = self.dt

        if(add_passive_scalar and not backward_u):
            if(self.step_num[None]+1 == self.final_ind):                
                passive_np = self.read_disk("s", self.final_dir, self.step_num[None]+1)
                self.passive.from_numpy(passive_np)
            else:
                self.disk_manage.read_disk_with_cache(self.passive,"s",self.passive_dir, self.step_num[None]+1)
                
        self.backtrack_flow_map(j,curr_dt,control_force_para)

        output_frame = False
        if j == 0 and output_image_frame:
            output_frame = True
            self.frame_idx[None]-=1
        if(write_gradient):
            scalar2vec(self.adj_u_x,self.adj_u_y,self.u_buffer)
            self.disk_manage.read_disk_with_cache(self.u_buffer,"adj_u",self.gradient_dir, self.step_num[None])  
            if(write_passive and add_passive_scalar):
                self.disk_manage.read_disk_with_cache(self.adj_passive,"adj_s",self.gradient_passive_dir, self.step_num[None])  
        print(self.step_num[None], self.final_condition())
        return  output_frame, self.step_num[None] == 0
        
    def backtrack_flow_map(self,j,curr_dt,control_force_para=None):

        if(add_control_force):
            gaussion_force_center,gaussion_force_radius, gaussion_force_strenth, gaussion_num = control_force_para["c"],control_force_para["r"],control_force_para["s"],control_force_para["n"]
            if(backward_u):                
                self.apply_gausion_force(self.control_force_x, self.control_force_y, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)
            self.calculate_adj_gausion_force(self.adj_u_x, self.adj_u_y, gaussion_force_center, gaussion_force_radius, gaussion_force_strenth, gaussion_num)

        self.disk_manage.read_disk_with_cache(self.u_buffer,"u",self.midpoint_dir, self.step_num[None])
        vec2scalar(self.mid_u_x,self.mid_u_y,self.u_buffer)

        if backward_u:
            calculate_nabla_u_w(self.u_x,self.u_y, self.adj_u_x,self.adj_u_y, self.adj_u_x_tem2,self.adj_u_y_tem2, self.X_x,self.X_y,self.dx)
        else:
            calculate_nabla_u_w(self.mid_u_x,self.mid_u_y, self.adj_u_x,self.adj_u_y, self.adj_u_x_tem2,self.adj_u_y_tem2, self.X_x,self.X_y,self.dx)
        
        if(add_passive_scalar):
            calculate_nabla_scalar_adjoint(self.passive, self.adj_passive, self.adj_passive_f_x, self.adj_passive_f_y, self.X_x, self.X_y, dx)

        if(use_short_BFECC):
            reset_to_identity(self.psi_tem, self.psi_x_tem, self.psi_y_tem, self.T_x_tem, self.T_y_tem,self.X,self.X_x,self.X_y)
            reset_to_identity(self.phi_tem, self.phi_x_tem, self.phi_y_tem, self.F_x_tem, self.F_y_tem,self.X,self.X_x,self.X_y)
            RK_grid(self.psi_x_tem, self.T_x_tem, self.mid_u_x, self.mid_u_y, self.dx,-curr_dt)
            RK_grid(self.psi_y_tem, self.T_y_tem, self.mid_u_x, self.mid_u_y, self.dx,-curr_dt)
            RK_grid(self.phi_x_tem, self.F_x_tem, self.mid_u_x, self.mid_u_y, self.dx,curr_dt)
            RK_grid(self.phi_y_tem, self.F_y_tem, self.mid_u_x, self.mid_u_y, self.dx,curr_dt)
            BFECC(
                self.adj_u_x, self.adj_u_y, self.adj_u_x_tem, self.adj_u_y_tem, 
                self.err_u_x, self.err_u_y, self.tmp_u_x, self.tmp_u_y, 
                self.T_x_tem, self.T_y_tem, self.psi_x_tem, self.psi_y_tem,
                self.F_x_tem, self.F_y_tem, self.phi_x_tem, self.phi_y_tem,
                dx, BFECC_clamp
            )
            add_fields(self.adj_u_x_tem,self.adj_u_x_tem2,self.adj_u_x,-2*curr_dt)
            add_fields(self.adj_u_y_tem,self.adj_u_y_tem2,self.adj_u_y,-2*curr_dt)

            if(add_passive_scalar):                
                RK_grid_only_psi(self.psi_tem, self.mid_u_x, self.mid_u_y, self.dx,-curr_dt)
                RK_grid_only_psi(self.phi_tem, self.mid_u_x, self.mid_u_y, self.dx, curr_dt)
                BFECC_scalar(self.adj_passive, self.final_passive,  self.err_passive, self.tmp_passive,  self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive,self.adj_passive)
                add_fields(self.adj_u_x,self.adj_passive_f_x,self.adj_u_x,-curr_dt)
                add_fields(self.adj_u_y,self.adj_passive_f_y,self.adj_u_y,-curr_dt)

            if backward_u:
                BFECC_scalar(self.passive, self.final_passive,  self.err_passive, self.tmp_passive,   self.psi_tem, self.phi_tem, dx, BFECC_clamp)
                copy_to(self.final_passive,self.passive)
                BFECC(
                    self.u_x, self.u_y, self.adj_u_x_tem, self.adj_u_y_tem, 
                    self.err_u_x, self.err_u_y, self.tmp_u_x, self.tmp_u_y, 
                    self.T_x_tem, self.T_y_tem, self.psi_x_tem, self.psi_y_tem,
                    self.F_x_tem, self.F_y_tem, self.phi_x_tem, self.phi_y_tem,
                    dx, BFECC_clamp
                )
                copy_to(self.adj_u_x_tem, self.u_x)
                copy_to(self.adj_u_y_tem, self.u_y)


        else:
            advect_u_grid(self.mid_u_x,self.mid_u_y, self.adj_u_x, self.adj_u_y, self.adj_u_x_tem, self.adj_u_y_tem, self.dx, -curr_dt, self.X_x,self.X_y)
            copy_to(self.adj_u_x_tem, self.adj_u_x)
            copy_to(self.adj_u_y_tem, self.adj_u_y)

            if backward_u:
                advect_scalar_grid(self.mid_u_x, self.mid_u_y, self.passive, self.final_passive,  self.dx, -curr_dt, self.X)
                copy_to(self.final_passive,self.passive)
                advect_u_grid(self.mid_u_x,self.mid_u_y, self.u_x, self.u_y, self.adj_u_x_tem, self.adj_u_y_tem, self.dx, -curr_dt, self.X_x,self.X_y)
                copy_to(self.adj_u_x_tem, self.u_x)
                copy_to(self.adj_u_y_tem, self.u_y)

            add_fields(self.adj_u_x,self.adj_u_x_tem2,self.adj_u_x,-curr_dt)
            add_fields(self.adj_u_y,self.adj_u_y_tem2,self.adj_u_y,-curr_dt)
    
            if(add_passive_scalar):
                advect_scalar_grid(self.mid_u_x, self.mid_u_y, self.adj_passive, self.final_passive,  self.dx, -curr_dt, self.X)
                copy_to(self.final_passive,self.adj_passive)
                add_fields(self.adj_u_x,self.adj_passive_f_x,self.adj_u_x,-curr_dt)
                add_fields(self.adj_u_y,self.adj_passive_f_y,self.adj_u_y,-curr_dt)
        
        self.march_psi_grid(curr_dt)
        #accumulate_init(self.adj_u_x_tem2, self.adj_u_y_tem2, self.init_adj_u_x,self.init_adj_u_y, self.T_x, self.T_y, self.psi_x, self.psi_y, dx, -2*curr_dt)
        accumulate_init_new(self.adj_u_x_tem2, self.adj_u_y_tem2, self.init_adj_u_x,self.init_adj_u_y, self.T_x, self.T_y, self.psi_x, self.psi_y, 
                            self.adj_u_x_AB_tem,self.adj_u_x_AB_tem2, self.adj_u_y_AB_tem, self.adj_u_y_AB_tem2,self.reinit_every-1-j,
                            dx, -2*curr_dt)
        
        if(add_passive_scalar):
            accumulate_init(self.adj_passive_f_x, self.adj_passive_f_y, self.init_adj_u_x,self.init_adj_u_y, self.T_x, self.T_y, self.psi_x, self.psi_y, dx, -curr_dt)
        
        self.solver.Poisson(self.adj_u_x, self.adj_u_y)
        if backward_u:
            if(add_control_force):
                add_fields(self.u_x,self.control_force_x,self.u_x,-curr_dt)
                add_fields(self.u_y,self.control_force_y,self.u_y,-curr_dt)
                accumulate_init(self.control_force_x, self.control_force_y, self.init_u_x,self.init_u_y, self.T_x, self.T_y, self.psi_x, self.psi_y, dx, -curr_dt)
            self.solver.Poisson(self.u_x, self.u_y)        

        if j == 0:
            self.disk_manage.read_disk_with_cache(self.phi,"phi", self.flowmap_phi_dir,int(self.step_num[None]/self.reinit_every))
            self.disk_manage.read_disk_with_cache(self.phi_xy_buffer,"phi_xy", self.flowmap_phi_xy_dir,int(self.step_num[None]/self.reinit_every))
            mat2vec(self.phi_x,self.phi_y,self.phi_xy_buffer)
            self.disk_manage.read_disk_with_cache(self.F_buffer,"F", self.flowmap_F_dir,int(self.step_num[None]/self.reinit_every))
            mat2vec(self.F_x,self.F_y,self.F_buffer)

            #self.disk_manage.read_disk_with_cache(self.psi,"psi", self.flowmap_psi_dir,int(self.step_num[None]/self.reinit_every))
            #self.disk_manage.read_disk_with_cache(self.phi_xy_buffer,"psi_xy", self.flowmap_psi_xy_dir,int(self.step_num[None]/self.reinit_every))
            #mat2vec(self.psi_x,self.psi_y,self.phi_xy_buffer)
            #self.disk_manage.read_disk_with_cache(self.F_buffer,"T", self.flowmap_T_dir,int(self.step_num[None]/self.reinit_every))
            #mat2vec(self.T_x,self.T_y,self.F_buffer)

            BFECC(
                self.init_adj_u_x, self.init_adj_u_y, self.adj_u_x, self.adj_u_y, 
                self.err_adj_u_x, self.err_adj_u_y, self.tmp_adj_u_x, self.tmp_adj_u_y, 
                self.F_x, self.F_y, self.phi_x, self.phi_y,
                self.T_x, self.T_y, self.psi_x, self.psi_y,                
                dx, BFECC_clamp
            )
            self.solver.Poisson(self.adj_u_x, self.adj_u_y)
            copy_to(self.adj_u_x, self.init_adj_u_x)
            copy_to(self.adj_u_y, self.init_adj_u_y)

            if backward_u:
                BFECC(
                    self.init_u_x, self.init_u_y, self.u_x, self.u_y, 
                    self.err_adj_u_x, self.err_adj_u_y, self.tmp_adj_u_x, self.tmp_adj_u_y, 
                    self.F_x, self.F_y, self.phi_x, self.phi_y,
                    self.T_x, self.T_y, self.psi_x, self.psi_y,                
                    dx, BFECC_clamp
                )
                self.solver.Poisson(self.u_x, self.u_y)
                copy_to(self.u_x, self.init_u_x)
                copy_to(self.u_y, self.init_u_y)

            if(add_passive_scalar):
                if(add_control_force):
                    BFECC_scalar(self.init_passive, self.passive,  self.err_passive, self.tmp_passive,    self.phi, self.psi,dx, BFECC_clamp)
                else:
                    self.disk_manage.read_disk_with_cache(self.passive,"s", self.passive_dir,self.step_num[None])
                copy_to(self.passive,self.init_passive)

                BFECC_scalar(self.init_adj_passive, self.adj_passive,  self.err_passive, self.tmp_passive,   self.phi, self.psi, dx, BFECC_clamp)
                if(not add_control_force or self.step_num[None] == 100 or self.step_num[None] == 200 or self.step_num[None] == 300):
                    if self.step_num[None] == 100:
                        target_density = np.load("density_R_arial_gradient.npy")
                        self.target_passive.from_numpy(target_density)
                    elif self.step_num[None] == 200:
                        target_density = np.load("density_A_arial_gradient.npy")
                        self.target_passive.from_numpy(target_density)
                    elif self.step_num[None] == 300:
                        target_density = np.load("density_P_arial_gradient.npy")
                        self.target_passive.from_numpy(target_density)
                    #self.adj_passive.fill(0.0)
                    add_fields(self.adj_passive,self.passive,self.adj_passive,1)
                    add_fields(self.adj_passive,self.target_passive,self.adj_passive,-1)
                    add_fields(self.target_passive,self.passive,self.target_passive,-1)
                    self.loss += np.sum((self.adj_passive.to_numpy() - target_density)**2)
                copy_to(self.adj_passive,self.init_adj_passive)

            reset_to_identity(self.psi, self.psi_x, self.psi_y, self.T_x, self.T_y,self.X,self.X_x,self.X_y)

if __name__ == '__main__':
    foward = False
    if(foward):
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
        simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir)
        simulator.init([-0.6,0.6,-0.6,0.6])    
        w_numpy = simulator.calculate_w()
        w_max = 15
        w_min = -15
        write_field(w_numpy, vortdir, from_frame, vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
        if(add_passive_scalar):
            passive_numpy = simulator.passive.to_numpy()
            write_field(passive_numpy, passivedir, from_frame, vmin=0, vmax=1, dpi=dpi_vor)
        last_output_substep = 0
        # Forward Step
        while True:
            output_frame, final_flag = simulator.forward_step_midpoint(True,True,True, True, True)
            if output_frame:
                w_numpy = simulator.calculate_w()     
                write_field(w_numpy, vortdir, simulator.frame_idx[None], vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    passive_numpy = simulator.passive.to_numpy()
                    write_field(passive_numpy, passivedir, simulator.frame_idx[None], vmin=0, vmax=1, dpi=dpi_vor)
                print("[Simulate] Finished frame: ", simulator.frame_idx[None], " in ", simulator.step_num[None]-last_output_substep, "substeps \n\n")
                last_output_substep = simulator.step_num[None]
            if final_flag:
                break
    else:
        logsdir = os.path.join('logs', exp_name)
        save_u_dir = "save_u"
        save_u_dir = os.path.join(logsdir, save_u_dir)
        simulator = LFM_Diff_Simulator(res_x,res_y,dx, act_dt, reinit_every,save_u_dir)
        simulator.init_gradient()
        gvortdir = 'gradient_vorticity'
        gvortdir = os.path.join(logsdir, gvortdir)
        os.makedirs(gvortdir, exist_ok=True)
        gpassivedir = 'gradient_passive'
        gpassivedir = os.path.join(logsdir, gpassivedir)
        os.makedirs(gpassivedir, exist_ok=True)
        w_max = 15#0.15#15*3
        w_min = -15#-0.15#-15*3
        # Backward Step
        while True:
            output_frame, final_flag = simulator.backtrack_step_midpoint(False,False)
            if output_frame:
                w_numpy = simulator.calculate_gw()     
                write_field(w_numpy, gvortdir, simulator.frame_idx[None], vmin=w_min,
                    vmax=w_max, dpi=dpi_vor)
                if(add_passive_scalar):
                    write_field(simulator.adj_passive.to_numpy(),gpassivedir, simulator.frame_idx[None], vmin=0,
                        vmax=1, dpi=dpi_vor)
            if final_flag:
                break

