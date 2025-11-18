from taichi_utils import *
from hyperparameters import *
##########################################################################
#################     Function Head   ###################################
##########################################################################
def RK_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), 
    dx:float, dt: float
):
    if(RK_number == 2):
        RK2_grid(psi_x, T_x, u_x0, u_y0, u_z0, dx, dt) 
    elif(RK_number == 4):
        RK4_grid(psi_x, T_x, u_x0, u_y0, u_z0, dx, dt)    


def RK_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), 
    dx:float,dt: float
):
    if(RK_number == 2):
        RK2_grid_only_psi(psi, u_x0, u_y0, u_z0, dx, dt)  
    elif(RK_number == 4):
        RK4_grid_only_psi(psi, u_x0, u_y0, u_z0, dx, dt)    

def advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), u_z2: ti.template(), 
    dx : float, dt : float,
    X_x:ti.template(), X_y:ti.template(), X_z:ti.template()
):
    if(RK_number == 2):
        RK2_advect_u_grid(u_x0, u_y0, u_z0, u_x1, u_y1, u_z1, u_x2, u_y2, u_z2, dx, dt, X_x, X_y, X_z)
    elif(RK_number == 4):
        RK4_advect_u_grid(u_x0, u_y0, u_z0, u_x1, u_y1, u_z1, u_x2, u_y2, u_z2, dx, dt, X_x, X_y, X_z)    

def advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    scalar1: ti.template(), 
    scalar2: ti.template(), 
    dx : float, dt : float,
    X:ti.template()
):
    if(RK_number == 2):
        RK2_advect_scalar_grid(u_x0, u_y0, u_z0, scalar1, scalar2, dx, dt, X)
    elif(RK_number == 4):
        RK4_advect_scalar_grid(u_x0, u_y0, u_z0, scalar1, scalar2, dx, dt, X)    

@ti.kernel
def advect_u(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), 
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(), 
    psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), dx: float
):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, tem1 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, tem1 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)
    # z velocity
    for I in ti.grouped(u_z1):
        u_at_psi, tem1 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = T_z[I].dot(u_at_psi)

@ti.kernel
def advect_scalar(
    scalar0: ti.template(),  
    scalar1: ti.template(),  
    psi: ti.template(), dx: float
):
    for I in ti.grouped(scalar0):
        scalar_at_psi = interp_2_scalar(scalar0,  psi[I], dx)
        scalar1[I] = scalar_at_psi

##########################################################################
#################     Implementation   ###################################
##########################################################################

@ti.kernel
def RK4_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), 
    dx:float, dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * neg_dt * dT_x_dt2 
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2 # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * neg_dt * u3
        T_x3 = T_x[I] + 1.0 * neg_dt * dT_x_dt3 # advance 1.0
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + neg_dt * 1./6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4) # advance full

@ti.kernel
def RK2_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(), 
    dx:float, dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * u2 
        T_x[I] = T_x[I] + neg_dt * dT_x_dt2 

@ti.kernel
def RK4_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    dx:float,dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi[I], dx)
        # prepare second
        psi1 = psi[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0,  u_z0, psi1, dx)
        # prepare third
        psi2 = psi[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        # third
        u3, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0,  u_z0, psi2, dx)
        # prepare fourth
        psi3 = psi[I] + 1.0 * neg_dt * u3
        # fourth
        u4, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi3, dx)
        # final advance
        psi[I] = psi[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)

@ti.kernel
def RK2_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    dx:float,dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi):
        # first
        u1, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi[I], dx)
        # prepare second
        psi1 = psi[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        # second
        u2, grad_u_at_psi = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi1, dx)
        # final advance
        psi[I] = psi[I] + neg_dt * u2

@ti.kernel
def advect_u2(
    u_x0: ti.template(), u_y0: ti.template(),  u_z0: ti.template(), 
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(), 
    psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(),
    dx: float
):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_x[I], dx)
        u_x1[I] = u_at_psi[0]
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_y[I], dx)
        u_y1[I] = u_at_psi[1]
    # z velocity
    for I in ti.grouped(u_z1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0, u_z0, psi_z[I], dx)
        u_z1[I] = u_at_psi[2]

@ti.kernel
def RK4_advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), u_z2: ti.template(), 
    dx : float, dt : float,
    X_x:ti.template(), X_y:ti.template(), X_z:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_x[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_x2[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_y[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_y2[I] = v5[1]

    for I in ti.grouped(u_z1):
        p1 = X_z[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_z2[I] = v5[2]

@ti.kernel
def RK2_advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), u_z1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), u_z2: ti.template(), 
    dx : float, dt : float,
    X_x:ti.template(), X_y:ti.template(), X_z:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_x[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p = p1 - v2 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_x2[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_y[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p = p1 -  v2 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_y2[I] = v5[1]

    for I in ti.grouped(u_z1):
        p1 = X_z[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p = p1 -  v2 * dt
        v5, _ = interp_u_MAC_grad(u_x1, u_y1, u_z1, p, dx)
        u_z2[I] = v5[2]

@ti.kernel
def RK4_advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    scalar1: ti.template(),  
    scalar2: ti.template(),  
    dx : float, dt : float,
    X:ti.template()
):
    for I in ti.grouped(scalar1):
        p1 = X[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        scalar_at_psi = interp_2_scalar(scalar1, p, dx)
        scalar2[I] = scalar_at_psi

@ti.kernel
def RK2_advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(), u_z0: ti.template(),
    scalar1: ti.template(),  
    scalar2: ti.template(),  
    dx : float, dt : float,
    X:ti.template()
):
    for I in ti.grouped(scalar1):
        p1 = X[I]
        v1, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _ = interp_u_MAC_grad(u_x0, u_y0, u_z0, p2, dx)
        p = p1 - v2 * dt
        scalar_at_psi = interp_2_scalar(scalar1, p, dx)
        scalar2[I] = scalar_at_psi


@ti.kernel
def clamp_u(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i,j,k in u:
        u1_l = sample(u1, i-1, j, k)
        u1_r = sample(u1, i+1, j, k)
        u1_b = sample(u1, i, j-1, k)
        u1_t = sample(u1, i, j+1, k)
        u1_a = sample(u1, i, j, k-1)
        u1_c = sample(u1, i, j, k+1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t, u1_a, u1_c)
        u2[i,j,k] = ti.math.clamp(u[i,j,k], mini, maxi)

@ti.kernel
def reset_to_identity(
    psi:ti.template(),
    psi_x: ti.template(), psi_y: ti.template(), psi_z: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), T_z: ti.template(),
    X:ti.template(),X_x:ti.template(),X_y:ti.template(),X_z:ti.template()
):
    for I in ti.grouped(psi):
        psi[I] = X[I]
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(psi_z):
        psi_z[I] = X_z[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(3, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(3, 1)
    for I in ti.grouped(T_z):
        T_z[I] = ti.Vector.unit(3, 2)

# TODO: substitute with weno
@ti.kernel
def calculate_nabla_u_w(
    u_x:ti.template(),u_y:ti.template(),u_z:ti.template(),
    w_x:ti.template(),w_y:ti.template(),w_z:ti.template(),
    r_x:ti.template(),r_y:ti.template(),r_z:ti.template(),
    X_x:ti.template(),X_y:ti.template(),X_z:ti.template(),
    dx:float
):
    for I in ti.grouped(u_x):
        tem0, grad_u_at_psi = interp_u_MAC_grad3(u_x, u_y, u_z,  X_x[I], dx)
        w, tem3 = interp_u_MAC_grad3(w_x, w_y, w_z,  X_x[I], dx)
        r_x[I] = (grad_u_at_psi.transpose()@w)[0]

    for I in ti.grouped(u_y):
        tem0, grad_u_at_psi = interp_u_MAC_grad3(u_x, u_y, u_z,  X_y[I], dx)
        w, tem3 = interp_u_MAC_grad3(w_x, w_y, w_z, X_y[I], dx)
        r_y[I] = (grad_u_at_psi.transpose()@w)[1]

    for I in ti.grouped(u_z):
        tem0, grad_u_at_psi = interp_u_MAC_grad3(u_x, u_y, u_z,  X_z[I], dx)
        w, tem3 = interp_u_MAC_grad3(w_x, w_y, w_z, X_z[I], dx)
        r_z[I] = (grad_u_at_psi.transpose()@w)[2]

@ti.kernel
def calculate_nabla_scalar_adjoint(
    scalar:ti.template(),
    adjoint_scalar:ti.template(),
    r_x:ti.template(), r_y:ti.template(), r_z:ti.template(),
    X_x:ti.template(), X_y:ti.template(), X_z:ti.template(),
    dx:float
):
    for I in ti.grouped(r_x):
        tem0, grad_scalar_at_psi = interp_grad_3(scalar,X_x[I],dx)
        grad_adjoint_scalar_at_psi, tem2= interp_grad_3(adjoint_scalar,X_x[I],dx)
        r_x[I] = grad_scalar_at_psi[0]*grad_adjoint_scalar_at_psi

    for I in ti.grouped(r_y):
        tem0, grad_scalar_at_psi = interp_grad_3(scalar,X_y[I],dx)
        grad_adjoint_scalar_at_psi, tem2= interp_grad_3(adjoint_scalar,X_y[I],dx)
        r_y[I] = grad_scalar_at_psi[1]*grad_adjoint_scalar_at_psi

    for I in ti.grouped(r_z):
        tem0, grad_scalar_at_psi= interp_grad_3(scalar,X_z[I],dx)
        grad_adjoint_scalar_at_psi, tem2 = interp_grad_3(adjoint_scalar,X_z[I],dx)
        r_z[I] = grad_scalar_at_psi[2]*grad_adjoint_scalar_at_psi

def BFECC(
    init_u_x, init_u_y, init_u_z, u_x, u_y, u_z, 
    err_u_x, err_u_y, err_u_z, tmp_u_x, tmp_u_y, tmp_u_z, 
    T_x, T_y, T_z, psi_x, psi_y, psi_z,
    F_x, F_y, F_z, phi_x, phi_y, phi_z,
    dx, BFECC_clamp
):
    advect_u(init_u_x, init_u_y, init_u_z, u_x, u_y, u_z, T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
    advect_u(u_x, u_y, u_z, err_u_x, err_u_y, err_u_z, F_x, F_y, F_z, phi_x, phi_y, phi_z, dx)                
    add_fields(err_u_x, init_u_x, err_u_x, -1.) 
    add_fields(err_u_y, init_u_y, err_u_y, -1.)
    add_fields(err_u_z, init_u_z, err_u_z, -1.)
    scale_field(err_u_x, 0.5, err_u_x)
    scale_field(err_u_y, 0.5, err_u_y)
    scale_field(err_u_z, 0.5, err_u_z)                
    advect_u(err_u_x, err_u_y, err_u_z, tmp_u_x, tmp_u_y, tmp_u_z, T_x, T_y, T_z,  psi_x, psi_y, psi_z, dx) 
    add_fields(u_x, tmp_u_x, err_u_x, -1.)
    add_fields(u_y, tmp_u_y, err_u_y, -1.)
    add_fields(u_z, tmp_u_z, err_u_z, -1.)
    if BFECC_clamp:
        clamp_u(err_u_x, u_x, tmp_u_x)
        clamp_u(err_u_y, u_y, tmp_u_y)
        clamp_u(err_u_z, u_z, tmp_u_z)
        copy_to(tmp_u_x, u_x)
        copy_to(tmp_u_y, u_y)
        copy_to(tmp_u_z, u_z)
    else:
        copy_to(err_u_x, u_x)
        copy_to(err_u_y, u_y)
        copy_to(err_u_z, u_z)

def BFECC_scalar(
    init_scalar, scalar, 
    err_scalar, tmp_scalar,  
    psi, phi,
    dx, BFECC_clamp
):
    advect_scalar(init_scalar, scalar, psi, dx)
    advect_scalar(scalar, err_scalar, phi, dx)                
    add_fields(err_scalar, init_scalar, err_scalar, -1.) 
    scale_field(err_scalar, 0.5, err_scalar)
    advect_scalar(err_scalar, tmp_scalar, psi, dx) 
    add_fields(scalar, tmp_scalar, err_scalar, -1.)
    if BFECC_clamp:
        clamp_u(err_scalar, scalar, tmp_scalar)
        copy_to(tmp_scalar, scalar)
    else:
        copy_to(err_scalar, scalar)

def BFECC2(
    init_u_x, init_u_y, init_u_z, u_x, u_y, u_z, 
    err_u_x, err_u_y, err_u_z, tmp_u_x, tmp_u_y, tmp_u_z, 
    T_x, T_y, T_z, psi_x, psi_y, psi_z,
    F_x, F_y, F_z, phi_x, phi_y, phi_z,
    dx, BFECC_clamp
):
    advect_u2(init_u_x, init_u_y, init_u_z, u_x, u_y, u_z, T_x, T_y, T_z, psi_x, psi_y, psi_z, dx)
    advect_u2(u_x, u_y, u_z, err_u_x, err_u_y, err_u_z, F_x, F_y, F_z, phi_x, phi_y, phi_z, dx)                
    add_fields(err_u_x, init_u_x, err_u_x, -1.) 
    add_fields(err_u_y, init_u_y, err_u_y, -1.)
    add_fields(err_u_z, init_u_z, err_u_z, -1.)
    scale_field(err_u_x, 0.5, err_u_x)
    scale_field(err_u_y, 0.5, err_u_y)                
    scale_field(err_u_z, 0.5, err_u_z)                
    advect_u2(err_u_x, err_u_y, err_u_z, tmp_u_x, tmp_u_y, tmp_u_z, T_x, T_y, T_z,  psi_x, psi_y, psi_z, dx) 
    add_fields(u_x, tmp_u_x, err_u_x, -1.)
    add_fields(u_y, tmp_u_y, err_u_y, -1.)
    add_fields(u_z, tmp_u_z, err_u_z, -1.)
    if BFECC_clamp:
        clamp_u(err_u_x, u_x, tmp_u_x)
        clamp_u(err_u_y, u_y, tmp_u_y)
        clamp_u(err_u_z, u_z, tmp_u_z)
        copy_to(tmp_u_x, u_x)
        copy_to(tmp_u_y, u_y)
        copy_to(tmp_u_z, u_z)
    else:
        copy_to(err_u_x, u_x)
        copy_to(err_u_y, u_y)
        copy_to(err_u_z, u_z)

@ti.kernel
def accumulate_init(
    f_x:ti.template(), f_y:ti.template(), f_z:ti.template(), init_u_x:ti.template(), init_u_y:ti.template(), init_u_z:ti.template(),
    F_x:ti.template(), F_y:ti.template(), F_z:ti.template(), phi_x:ti.template(), phi_y:ti.template(), phi_z:ti.template(),
    dx:float, dt:float
):
    for I in ti.grouped(init_u_x):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z,  phi_x[I], dx)
        init_u_x[I]+=F_x[I]@v*dt 

    for I in ti.grouped(init_u_y):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z,  phi_y[I], dx)
        init_u_y[I]+=F_y[I]@v*dt 

    for I in ti.grouped(init_u_z):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z,  phi_z[I], dx)
        init_u_z[I]+=F_z[I]@v*dt 

@ti.kernel
def accumulate_init_new(
    f_x:ti.template(), f_y:ti.template(), f_z:ti.template(),  init_u_x:ti.template(), init_u_y:ti.template(), init_u_z:ti.template(),
    F_x:ti.template(), F_y:ti.template(), F_z:ti.template(), phi_x:ti.template(), phi_y:ti.template(), phi_z:ti.template(),
    H_x_prev:ti.template(), H_x_prev_prev:ti.template(),
    H_y_prev:ti.template(), H_y_prev_prev:ti.template(),
    H_z_prev:ti.template(), H_z_prev_prev:ti.template(),
    ind:int,
    dx:float, dt:float
):
    for I in ti.grouped(init_u_x):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z,  phi_x[I], dx)
        if(ind == 0):
            init_u_x[I]+=F_x[I]@v*dt 
        elif(ind == 1):
            init_u_x[I]+=(23*F_x[I]@v-16*F_x[I]@v+5*H_x_prev[I])/12*dt 
        else:
            init_u_x[I]+=(23*F_x[I]@v-16*H_x_prev[I]+5*H_x_prev_prev[I])/12*dt 
        H_x_prev_prev[I] = H_x_prev[I]
        H_x_prev[I] = F_x[I]@v


    for I in ti.grouped(init_u_y):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z, phi_y[I], dx)
        if(ind == 0):
            init_u_y[I]+=F_y[I]@v*dt 
        elif(ind == 1):
            init_u_y[I]+=(23*F_y[I]@v-16*F_y[I]@v+5*H_y_prev[I])/12*dt 
        else:
            init_u_y[I]+=(23*F_y[I]@v-16*H_y_prev[I]+5*H_y_prev_prev[I])/12*dt 
        H_y_prev_prev[I] = H_y_prev[I]
        H_y_prev[I] = F_y[I]@v

    for I in ti.grouped(init_u_z):
        v, tem3= interp_u_MAC_grad(f_x, f_y, f_z, phi_z[I], dx)
        if(ind == 0):
            init_u_z[I]+=F_z[I]@v*dt 
        elif(ind == 1):
            init_u_z[I]+=(23*F_z[I]@v-16*F_z[I]@v+5*H_z_prev[I])/12*dt 
        else:
            init_u_z[I]+=(23*F_z[I]@v-16*H_z_prev[I]+5*H_z_prev_prev[I])/12*dt 
        H_z_prev_prev[I] = H_z_prev[I]
        H_z_prev[I] = F_z[I]@v
