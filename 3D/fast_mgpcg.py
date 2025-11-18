import torch
import numpy as np
import time
import taichi as ti
from poisson_solver_permenant_stream import PoissonSolver
from hyperparameters import *
from taichi_utils import *

@ti.data_oriented
class MGPCG_3:
    def __init__(self, boundary_types, N, base_level=3):
        self.res_x,self.res_y,self.res_z = N[0],N[1],N[2]
        self.dx = 1/self.res_y        
        self.tile_dim_vec = [self.res_x//8, self.res_y//8, self.res_z//8]

        self.p = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.u_div = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.a_diag = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.a_x = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.a_y = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.a_z = ti.field(ti.f32, shape=(self.res_x, self.res_y, self.res_z))
        self.is_dof = ti.field(ti.u8, shape=(self.res_x, self.res_y, self.res_z))

        self.b_pretorch = ti.field(ti.f32, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))
        self.a_diag_pretorch = ti.field(ti.f32, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))
        self.a_x_pretorch = ti.field(ti.f32, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))
        self.a_y_pretorch = ti.field(ti.f32, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))
        self.a_z_pretorch = ti.field(ti.f32, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))
        self.is_dof_pretorch = ti.field(ti.u8, shape=(self.tile_dim_vec[0]*8, self.tile_dim_vec[1]*8, self.tile_dim_vec[2]*8))

        self.dim_x, self.dim_y, self.dim_z = self.tile_dim_vec[0] * 8, self.tile_dim_vec[1] * 8, self.tile_dim_vec[2] * 8

        self.xinitial_pytorch = torch.zeros((self.dim_x, self.dim_y, self.dim_z), dtype=torch.float32, device='cuda')
        self.boundary_types = boundary_types
        self.init()

    @ti.kernel
    def divergence(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template(), u_div:ti.template()):
        u_dim, v_dim, w_dim = u_div.shape
        for i, j, k in u_div:
            vl = sample(u_x, i, j, k)
            vr = sample(u_x, i + 1, j, k)
            vb = sample(u_y, i, j, k)
            vt = sample(u_y, i, j + 1, k)
            va = sample(u_z, i, j, k)
            vc = sample(u_z, i, j, k + 1)
            u_div[i,j,k] = vr - vl + vt - vb + vc - va

    @ti.kernel
    def construct_adiag(self, a_diag:ti.template()):
        udim, vdim, wdim = a_diag.shape
        for i,j,k in a_diag:
            num = 6.0
            if i == 0 or i == udim - 1:
                num += -1.0
            if j == 0 or j == vdim - 1:
                num += -1.0
            if k == 0 or k == wdim - 1:
                num += -1.0
            a_diag[i,j,k] = num

    @ti.kernel
    def construct_ax(self, a_x:ti.template(), axis:int):
        udim, vdim, wdim = a_x.shape
        for I in ti.grouped(a_x):
            num = -1.0
            if axis == 0:
                if I[axis] == udim - 1:
                    num = 0.0
            elif axis == 1:
                if I[axis] == vdim - 1:
                    num = 0.0
            elif axis == 2:
                if I[axis] == wdim - 1:
                    num = 0.0
            a_x[I] = num


    @ti.kernel
    def extend_to_pretorch(self, ti_field:ti.template(), pretorch:ti.template()):
        for i,j,k in ti_field:
            pretorch[i,j,k] = ti_field[i,j,k]

    @ti.kernel
    def extend_boundary_field_reverse(self, a:ti.template(),b:ti.template()):
        shape_x,shape_y,shape_z=a.shape
        for i,j,k in b:
            if(i<shape_x and j<shape_y and k<shape_z):
                b[i,j,k]=a[i,j,k]
            else:
                b[i,j,k]=0

    @ti.kernel
    def subtract_grad_p(self, p:ti.template(), u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = p.shape
        for i, j, k in u_x:
            pr = sample(p, i, j, k)
            pl = sample(p, i-1, j, k)
            if i-1 < 0:
                pl = 0
            if i >= u_dim:
                pr = 0
            u_x[i,j,k] -= (pr - pl)
        for i, j, k in u_y:
            pt = sample(p, i, j, k)
            pb = sample(p, i, j-1, k)
            if j-1 < 0:
                pb = 0
            if j >= v_dim:
                pt = 0
            u_y[i,j,k] -= pt - pb
        for i, j, k in u_z:
            pc = sample(p, i, j, k)
            pa = sample(p, i, j, k-1)
            if k-1 < 0:
                pa = 0
            if k >= w_dim:
                pc = 0
            u_z[i,j,k] -= pc - pa

    def rearrange_tensor_for_cuda(self, tensor, tile_dim_vec, tile_size=8):
        # Assume tensor is on GPU
        # Calculate padding sizes to ensure dimensions are divisible by tile_size
        
        # Reshape into tiles
        tensor_reshaped = tensor.view(
            tile_dim_vec[0], tile_size,
            tile_dim_vec[1], tile_size,
            tile_dim_vec[2], tile_size
        )
        # print(tensor_reshaped.shape)
        
        # Rearrange axes to bring tiles together
        tensor_reordered = tensor_reshaped.permute(0, 2, 4, 1, 3, 5).contiguous()
        # tensor_reordered = tensor_reshaped.contiguous()
        # print(tensor_reordered.shape)
        # Now tensor_reordered has shape (tile_dim_x, tile_dim_y, tile_dim_z, tile_size, tile_size, tile_size)
        # Flatten the last three axes (voxels within a tile)
        tensor_tiles = tensor_reordered.view(-1, tile_size ** 3)
        
        # print(tensor_tiles.shape)
        
        # Flatten the entire tensor to match the CUDA kernel's expected 1D layout
        tensor_flat = tensor_tiles.view(-1)
        
        return tensor_flat


    def rearrange_tensor_from_cuda(self, tensor_flat, original_shape, tile_dim_vec, tile_size=8):
        tile_num = tile_dim_vec[0] * tile_dim_vec[1] * tile_dim_vec[2]
        tensor = tensor_flat.view(tile_num, tile_size ** 3)
        tensor = tensor.view(
            tile_dim_vec[0], tile_dim_vec[1], tile_dim_vec[2],
            tile_size, tile_size, tile_size
        )
        tensor = tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
        tensor = tensor.view(original_shape)
        return tensor


    def init(self):    
        self.a_x.fill(0.0)
        self.a_y.fill(0.0)
        self.a_z.fill(0.0)
        self.a_diag.fill(0.0)
        self.is_dof.fill(1)

        self.construct_ax(self.a_x, 0)
        self.construct_ax(self.a_y, 1)
        self.construct_ax(self.a_z, 2)
        self.construct_adiag(self.a_diag)

        self.extend_boundary_field_reverse(self.is_dof, self.is_dof_pretorch)
        self.extend_to_pretorch(self.a_x, self.a_x_pretorch)
        self.extend_to_pretorch(self.a_y, self.a_y_pretorch)
        self.extend_to_pretorch(self.a_z, self.a_z_pretorch)
        self.extend_to_pretorch(self.a_diag, self.a_diag_pretorch)

        self.a_diag_pytorch = self.a_diag_pretorch.to_torch(device="cuda:0")
        self.a_x_pytorch = self.a_x_pretorch.to_torch(device="cuda:0")
        self.a_y_pytorch = self.a_y_pretorch.to_torch(device="cuda:0")
        self.a_z_pytorch = self.a_z_pretorch.to_torch(device="cuda:0")
        self.is_dof_pytorch = self.is_dof_pretorch.to_torch(device="cuda:0")
        self.tile_size = 8

        self.a_diag_pytorch_flat = self.rearrange_tensor_for_cuda(self.a_diag_pytorch, self.tile_dim_vec, self.tile_size)
        self.a_x_pytorch_flat = self.rearrange_tensor_for_cuda(self.a_x_pytorch, self.tile_dim_vec, self.tile_size)
        self.a_y_pytorch_flat = self.rearrange_tensor_for_cuda(self.a_y_pytorch, self.tile_dim_vec, self.tile_size)
        self.a_z_pytorch_flat = self.rearrange_tensor_for_cuda(self.a_z_pytorch, self.tile_dim_vec, self.tile_size)
        self.is_dof_pytorch_flat = self.rearrange_tensor_for_cuda(self.is_dof_pytorch, self.tile_dim_vec, self.tile_size)

        self.solver_poisson = PoissonSolver(self.tile_dim_vec, level_num=3, bottom_smoothing=40)
        self.is_uniform = False
        self.uniform_coef = 1.0
        # 1 is neumann
        self.neg_bc_type_vec = [1,1,1]
        self.pos_bc_type_vec = [1,1,1]
    
    @ti.kernel
    def apply_bc(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0,0] == 2:
                u_x[i,j,k] = 0.0
            if i == u_dim - 1 and self.boundary_types[0,1] == 2:
                u_x[i,j,k] = 0.0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1,0] == 2:
                u_y[i,j,k] = 0
            if j == v_dim - 1 and self.boundary_types[1,1] == 2:
                u_y[i,j,k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2,0] == 2:
                u_z[i,j,k] = 0
            if k == w_dim - 1 and self.boundary_types[2,1] == 2:
                u_z[i,j,k] = 0

    @ti.kernel
    def apply_bc_adj(self, u_x: ti.template(), u_y: ti.template(), u_z: ti.template()):
        u_dim, v_dim, w_dim = u_x.shape
        for i, j, k in u_x:
            if i == 0 and self.boundary_types[0,0] == 2:
                u_x[i,j,k] = 0
            if i == u_dim - 1 and self.boundary_types[0,1] == 2:
                u_x[i,j,k] = 0
        u_dim, v_dim, w_dim = u_y.shape
        for i, j, k in u_y:
            if j == 0 and self.boundary_types[1,0] == 2:
                u_y[i,j,k] = 0
            if j == v_dim - 1 and self.boundary_types[1,1] == 2:
                u_y[i,j,k] = 0
        u_dim, v_dim, w_dim = u_z.shape
        for i, j, k in u_z:
            if k == 0 and self.boundary_types[2,0] == 2:
                u_z[i,j,k] = 0
            if k == w_dim - 1 and self.boundary_types[2,1] == 2:
                u_z[i,j,k] = 0

    def Poisson(self, u_x, u_y, u_z, verbose = False):
        self.init()
        
        self.apply_bc(u_x,u_y,u_z)
        self.u_div.fill(0.0)
        self.divergence(u_x, u_y, u_z, self.u_div)
        if poisson_output_log:
            div_np = self.u_div.to_numpy()
            print("divergence before projection: ", np.max(div_np))

        scale_field(self.u_div, -1.0, self.u_div)
        self.extend_to_pretorch(self.u_div, self.b_pretorch)

        self.b_pytorch = self.b_pretorch.to_torch(device="cuda:0")
        self.b_pytorch_flat = self.rearrange_tensor_for_cuda(self.b_pytorch, self.tile_dim_vec, self.tile_size)
        self.xinitial_pytorch_flat = torch.zeros_like(self.b_pytorch_flat, device='cuda').contiguous()

        self.solver_poisson.setup(
            a_x=self.a_x_pytorch_flat,
            a_y=self.a_y_pytorch_flat,
            a_z=self.a_z_pytorch_flat,
            b=self.b_pytorch_flat,
            a_diag=self.a_diag_pytorch_flat,
            is_dof=self.is_dof_pytorch_flat,
            x_initial=self.xinitial_pytorch_flat,
            is_uniform=self.is_uniform,
            uniform_coef=self.uniform_coef,
            neg_bc_type_vec=self.neg_bc_type_vec,
            pos_bc_type_vec=self.pos_bc_type_vec
        )

        self.solver_poisson.build()
        self.solver_poisson.solve(self.xinitial_pytorch_flat)
        self.result_tensor = self.rearrange_tensor_from_cuda(self.xinitial_pytorch_flat, (self.res_x, self.res_y, self.res_z), self.tile_dim_vec, self.tile_size)
        self.p.from_torch(self.result_tensor)
        self.subtract_grad_p(self.p, u_x, u_y, u_z)
        self.apply_bc(u_x,u_y,u_z)
        if poisson_output_log:            
            self.divergence(u_x, u_y, u_z, self.u_div)
            div_np = self.u_div.to_numpy()
            print("divergence after projection: ", np.max(div_np))

    def Poisson_adj(self, u_x, u_y, u_z, verbose = False):
        self.init()
        
        self.apply_bc_adj(u_x,u_y,u_z)
        self.u_div.fill(0.0)
        self.divergence(u_x, u_y, u_z, self.u_div)
        if poisson_output_log:
            div_np = self.u_div.to_numpy()
            print("divergence before projection: ", np.max(div_np))

        scale_field(self.u_div, -1.0, self.u_div)
        self.extend_to_pretorch(self.u_div, self.b_pretorch)

        self.b_pytorch = self.b_pretorch.to_torch(device="cuda:0")
        self.b_pytorch_flat = self.rearrange_tensor_for_cuda(self.b_pytorch, self.tile_dim_vec, self.tile_size)
        self.xinitial_pytorch_flat = torch.zeros_like(self.b_pytorch_flat, device='cuda').contiguous()

        self.solver_poisson.setup(
            a_x=self.a_x_pytorch_flat,
            a_y=self.a_y_pytorch_flat,
            a_z=self.a_z_pytorch_flat,
            b=self.b_pytorch_flat,
            a_diag=self.a_diag_pytorch_flat,
            is_dof=self.is_dof_pytorch_flat,
            x_initial=self.xinitial_pytorch_flat,
            is_uniform=self.is_uniform,
            uniform_coef=self.uniform_coef,
            neg_bc_type_vec=self.neg_bc_type_vec,
            pos_bc_type_vec=self.pos_bc_type_vec
        )

        self.solver_poisson.build()
        self.solver_poisson.solve(self.xinitial_pytorch_flat)
        self.result_tensor = self.rearrange_tensor_from_cuda(self.xinitial_pytorch_flat, (self.res_x, self.res_y, self.res_z), self.tile_dim_vec, self.tile_size)
        self.p.from_torch(self.result_tensor)
        self.subtract_grad_p(self.p, u_x, u_y, u_z)
        self.apply_bc_adj(u_x,u_y,u_z)
        if poisson_output_log:            
            self.divergence(u_x, u_y, u_z, self.u_div)
            div_np = self.u_div.to_numpy()
            print("divergence after projection: ", np.max(div_np))

if __name__ == '__main__':
    ti.init(arch=ti.cuda, debug = False,default_fp = ti.f32)

    from init_conditions import *
    
    u = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
    w = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
    p = ti.field(ti.f32, shape=(res_x, res_y, res_z))
    u_x = ti.field(ti.f32, shape=(res_x + 1, res_y, res_z))
    u_y = ti.field(ti.f32, shape=(res_x, res_y + 1, res_z))
    u_z = ti.field(ti.f32, shape=(res_x, res_y, res_z + 1))
    
    smoke1 = ti.field(ti.f32, shape=(res_x, res_y, res_z))
    smoke2 = ti.field(ti.f32, shape=(res_x, res_y, res_z))

    X = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z))
    X_x = ti.Vector.field(3, ti.f32, shape=(res_x + 1, res_y, res_z))
    X_y = ti.Vector.field(3, ti.f32, shape=(res_x, res_y + 1, res_z))
    X_z = ti.Vector.field(3, ti.f32, shape=(res_x, res_y, res_z + 1))
    center_coords_func(X, dx)
    x_coords_func(X_x, dx)
    y_coords_func(X_y, dx)
    z_coords_func(X_z, dx)

    init_vorts_leapfrog(X, u, smoke1, smoke2)
    split_central_vector(u, u_x, u_y, u_z)

    solver = MGPCG_3(None, N = [res_x, res_y, res_z])
    solver.Poisson(u_x, u_y, u_z, verbose = True)

























