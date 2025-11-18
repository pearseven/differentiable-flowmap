from taichi_utils import *
from hyperparameters import *
##########################################################################
#################     Function Head   ###################################
##########################################################################
def RK_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float, dt: float
):
    if(RK_number == 2):
        RK2_grid(psi_x, T_x, u_x0, u_y0, dx, dt) 
    elif(RK_number == 4):
        RK4_grid(psi_x, T_x, u_x0, u_y0, dx, dt)    


def RK_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float,dt: float
):
    if(RK_number == 2):
        RK2_grid_only_psi(psi, u_x0, u_y0, dx, dt)  
    elif(RK_number == 4):
        RK4_grid_only_psi(psi, u_x0, u_y0, dx, dt)    

def advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), 
    dx : float, dt : float,
    X_horizontal:ti.template(), X_vertical:ti.template()
):
    if(RK_number == 2):
        RK2_advect_u_grid(u_x0, u_y0, u_x1, u_y1, u_x2, u_y2, dx, dt, X_horizontal, X_vertical)
    elif(RK_number == 4):
        RK4_advect_u_grid(u_x0, u_y0, u_x1, u_y1, u_x2, u_y2, dx, dt, X_horizontal, X_vertical)    

def advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    scalar1: ti.template(), 
    scalar2: ti.template(), 
    dx : float, dt : float,
    X:ti.template()
):
    if(RK_number == 2):
        RK2_advect_scalar_grid(u_x0, u_y0, scalar1, scalar2, dx, dt, X)
    elif(RK_number == 4):
        RK4_advect_scalar_grid(u_x0, u_y0, scalar1, scalar2, dx, dt, X)    

@ti.kernel
def advect_u(
    u_x0: ti.template(), u_y0: ti.template(), 
    u_x1: ti.template(), u_y1: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), 
    psi_x: ti.template(), psi_y: ti.template(), dx: float
):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0,  psi_x[I], dx)
        u_x1[I] = T_x[I].dot(u_at_psi)
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0,  psi_y[I], dx)
        u_y1[I] = T_y[I].dot(u_at_psi)

@ti.kernel
def advect_scalar(
    scalar0: ti.template(),  
    scalar1: ti.template(),  
    psi: ti.template(), dx: float
):
    for I in ti.grouped(scalar0):
        scalar_at_psi = interp_2_scalar(scalar0,  psi[I], dx)
        scalar1[I] = scalar_at_psi

@ti.kernel
def advect_gradient_neuralfluid(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), 
    dx : float, dt : float,
    X_horizontal:ti.template(), X_vertical:ti.template()
):
    pass



##########################################################################
#################     Implementation   ###################################
##########################################################################

@ti.kernel
def RK4_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float, dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # prepare third
        psi_x2 = psi_x[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        T_x2 = T_x[I] + 0.5 * neg_dt * dT_x_dt2 
        # third
        u3, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0,  psi_x2, dx)
        dT_x_dt3 = grad_u_at_psi @ T_x2 # time derivative of T
        # prepare fourth
        psi_x3 = psi_x[I] + 1.0 * neg_dt * u3
        T_x3 = T_x[I] + 1.0 * neg_dt * dT_x_dt3 # advance 1.0
        # fourth
        u4, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi_x3, dx)
        dT_x_dt4 = grad_u_at_psi @ T_x3 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)
        T_x[I] = T_x[I] + neg_dt * 1./6 * (dT_x_dt1 + 2 * dT_x_dt2 + 2 * dT_x_dt3 + dT_x_dt4) # advance full

@ti.kernel
def RK2_grid(
    psi_x: ti.template(), T_x: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float, dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi_x):
        # first
        u1, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi_x[I], dx)
        dT_x_dt1 = grad_u_at_psi @ T_x[I] # time derivative of T
        # prepare second
        psi_x1 = psi_x[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        T_x1 = T_x[I] + 0.5 * neg_dt * dT_x_dt1
        # second
        u2, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi_x1, dx)
        dT_x_dt2 = grad_u_at_psi @ T_x1 # time derivative of T
        # final advance
        psi_x[I] = psi_x[I] + neg_dt * u2 
        T_x[I] = T_x[I] + neg_dt * dT_x_dt2 

@ti.kernel
def RK4_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float,dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi):
        # first
        u1, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi[I], dx)
        # prepare second
        psi1 = psi[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        # second
        u2, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0,  psi1, dx)
        # prepare third
        psi2 = psi[I] + 0.5 * neg_dt * u2 # advance 0.5 again
        # third
        u3, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0,  psi2, dx)
        # prepare fourth
        psi3 = psi[I] + 1.0 * neg_dt * u3
        # fourth
        u4, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi3, dx)
        # final advance
        psi[I] = psi[I] + neg_dt * 1./6 * (u1 + 2 * u2 + 2 * u3 + u4)

@ti.kernel
def RK2_grid_only_psi(
    psi: ti.template(), 
    u_x0: ti.template(), u_y0: ti.template(), dx:float,dt: float
):
    neg_dt = -1 * dt # travel back in time
    for I in ti.grouped(psi):
        # first
        u1, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0, psi[I], dx)
        # prepare second
        psi1 = psi[I] + 0.5 * neg_dt * u1 # advance 0.5 steps
        # second
        u2, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad(u_x0, u_y0,  psi1, dx)
        # final advance
        psi[I] = psi[I] + neg_dt * u2

@ti.kernel
def advect_u2(
    u_x0: ti.template(), u_y0: ti.template(), 
    u_x1: ti.template(), u_y1: ti.template(), 
    T_x: ti.template(), T_y: ti.template(), 
    psi_x: ti.template(), psi_y: ti.template(), dx: float
):
    # x velocity
    for I in ti.grouped(u_x1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0,  psi_x[I], dx)
        u_x1[I] = u_at_psi[0]
    # y velocity
    for I in ti.grouped(u_y1):
        u_at_psi, tem1,tem2,tem3 = interp_u_MAC_grad(u_x0, u_y0,  psi_y[I], dx)
        u_y1[I] = u_at_psi[1]

@ti.kernel
def RK4_advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), 
    dx : float, dt : float,
    X_horizontal:ti.template(), X_vertical:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x1, u_y1, p, dx)
        u_x2[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x1, u_y1, p, dx)
        u_y2[I] = v5[1]

@ti.kernel
def RK2_advect_u_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    u_x1: ti.template(), u_y1: ti.template(), 
    u_x2: ti.template(), u_y2: ti.template(), 
    dx : float, dt : float,
    X_horizontal:ti.template(), X_vertical:ti.template()
):
    for I in ti.grouped(u_x1):
        p1 = X_horizontal[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p = p1 - v2 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x1, u_y1, p, dx)
        u_x2[I] = v5[0]
        
    for I in ti.grouped(u_y1):
        p1 = X_vertical[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p = p1 -  v2 * dt
        v5, _, _, _ = interp_u_MAC_grad(u_x1, u_y1, p, dx)
        u_y2[I] = v5[1]

@ti.kernel
def RK4_advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    scalar1: ti.template(),  
    scalar2: ti.template(),  
    dx : float, dt : float,
    X:ti.template()
):
    for I in ti.grouped(scalar1):
        p1 = X[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p3 = p1 - v2 * dt * 0.5
        v3, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p3, dx)
        p4 = p1 - v3 * dt
        v4, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p4, dx)
        p = p1 - (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * dt
        scalar_at_psi = interp_2_scalar(scalar1, p, dx)
        scalar2[I] = scalar_at_psi

@ti.kernel
def RK2_advect_scalar_grid(
    u_x0: ti.template(), u_y0: ti.template(),
    scalar1: ti.template(),  
    scalar2: ti.template(),  
    dx : float, dt : float,
    X:ti.template()
):
    for I in ti.grouped(scalar1):
        p1 = X[I]
        v1, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p1, dx)
        p2 = p1 - v1 * dt * 0.5
        v2, _, _, _ = interp_u_MAC_grad(u_x0, u_y0, p2, dx)
        p = p1 - v2 * dt
        scalar_at_psi = interp_2_scalar(scalar1, p, dx)
        scalar2[I] = scalar_at_psi


@ti.kernel
def clamp_u(u: ti.template(), u1: ti.template(), u2: ti.template()):
    for i,j in u:
        u1_l = sample(u1, i-1, j)
        u1_r = sample(u1, i+1, j)
        u1_b = sample(u1, i, j-1)
        u1_t = sample(u1, i, j+1)
        maxi = ti.math.max(u1_l, u1_r, u1_b, u1_t)
        mini = ti.math.min(u1_l, u1_r, u1_b, u1_t)
        u2[i,j] = ti.math.clamp(u[i,j], mini, maxi)

@ti.kernel
def reset_to_identity(
    psi:ti.template(),psi_x: ti.template(), psi_y: ti.template(), 
    T_x: ti.template(), T_y: ti.template(),
    X:ti.template(),X_x:ti.template(),X_y:ti.template()
):
    for I in ti.grouped(psi):
        psi[I] = X[I]
    for I in ti.grouped(psi_x):
        psi_x[I] = X_x[I]
    for I in ti.grouped(psi_y):
        psi_y[I] = X_y[I]
    for I in ti.grouped(T_x):
        T_x[I] = ti.Vector.unit(2, 0)
    for I in ti.grouped(T_y):
        T_y[I] = ti.Vector.unit(2, 1)

# TODO: substitute with weno
@ti.kernel
def calculate_nabla_u_w(
    u_x:ti.template(),u_y:ti.template(),
    w_x:ti.template(),w_y:ti.template(),
    r_x:ti.template(),r_y:ti.template(),
    X_x:ti.template(),X_y:ti.template(),
    dx:float
):
    for I in ti.grouped(u_x):
        tem0, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad3(u_x, u_y,  X_x[I], dx)
        w, tem3,tem1,tem2 = interp_u_MAC_grad3(w_x, w_y,  X_x[I], dx)
        r_x[I] = (grad_u_at_psi.transpose()@w)[0]

    for I in ti.grouped(u_y):
        tem0, grad_u_at_psi,tem1,tem2 = interp_u_MAC_grad3(u_x, u_y,  X_y[I], dx)
        w, tem3,tem1,tem2 = interp_u_MAC_grad3(w_x, w_y,  X_y[I], dx)
        r_y[I] = (grad_u_at_psi.transpose()@w)[1]

@ti.kernel
def calculate_nabla_scalar_adjoint(
    scalar:ti.template(),
    adjoint_scalar:ti.template(),
    r_x:ti.template(), r_y:ti.template(),
    X_x:ti.template(), X_y:ti.template(),
    dx:float
):
    for I in ti.grouped(r_x):
        tem0, grad_scalar_at_psi,tem1 = interp_grad_3(scalar,X_x[I],dx)
        grad_adjoint_scalar_at_psi, tem2,tem1 = interp_grad_3(adjoint_scalar,X_x[I],dx)
        r_x[I] = grad_scalar_at_psi[0]*grad_adjoint_scalar_at_psi

    for I in ti.grouped(r_y):
        tem0, grad_scalar_at_psi,tem1 = interp_grad_3(scalar,X_y[I],dx)
        grad_adjoint_scalar_at_psi, tem2,tem1 = interp_grad_3(adjoint_scalar,X_y[I],dx)
        r_y[I] = grad_scalar_at_psi[1]*grad_adjoint_scalar_at_psi

# TODO: substitute with weno
@ti.kernel
def calculate_nabla_u_w_test(
    u_x:ti.template(),u_y:ti.template(),
    u2:ti.template(),
    w_x:ti.template(),w_y:ti.template(),
    r_x:ti.template(),r_y:ti.template(),
    X_x:ti.template(),X_y:ti.template(),
    dx:float
):
    for i,j in u2:
        u2[i,j] = ((u_x[i,j]+u_x[i+1,j])/2)**2+((u_y[i,j]+u_y[i,j+1])/2)**2
        #u2[i,j] = u2[i,j]/2

    for i,j in u_x:
        r_x[i,j] = (sample(u2,i,j)-sample(u2,i-1,j))/dx

    for i,j in u_y:
        r_y[i,j] = (sample(u2,i,j)-sample(u2,i,j-1))/dx

@ti.kernel
def calculate_nabla_u_w_test2(
    u_x:ti.template(),u_y:ti.template(),
    w_x:ti.template(),w_y:ti.template(),
    r_x:ti.template(),r_y:ti.template(),
    X_x:ti.template(),X_y:ti.template(),
    dx:float
):

    for i,j in u_x:
        r_x[i,j] = w_x[i,j]*(
            sample(u_x,i+1,j)-sample(u_x,i-1,j)
        )/dx/2+ (
            (sample(u_y,i,j+1)+sample(u_y,i,j))/2-(sample(u_y,i-1,j+1)+sample(u_y,i-1,j))/2
        )/dx*(sample(w_y,i-1,j)+sample(w_y,i,j)+sample(w_y,i-1,j+1)+sample(w_y,i,j+1))/4

    for i,j in u_y:
        r_y[i,j] = w_y[i,j]*(
            sample(u_y,i,j+1)-sample(u_y,i,j-1)
        )/dx/2+ (
            (sample(u_x,i+1,j)+sample(u_x,i,j))/2-(sample(u_x,i+1,j-1)+sample(u_x,i,j-1))/2
        )/dx*(sample(w_x,i,j-1)+sample(w_x,i,j)+sample(w_x,i+1,j-1)+sample(w_x,i+1,j))/4

def BFECC(
    init_u_x, init_u_y, u_x, u_y, 
    err_u_x, err_u_y, tmp_u_x, tmp_u_y, 
    T_x, T_y, psi_x, psi_y,
    F_x, F_y, phi_x, phi_y,
    dx, BFECC_clamp
):
    advect_u(init_u_x, init_u_y, u_x, u_y, T_x, T_y, psi_x, psi_y, dx)
    advect_u(u_x, u_y, err_u_x, err_u_y,F_x, F_y, phi_x, phi_y, dx)                
    add_fields(err_u_x, init_u_x, err_u_x, -1.) 
    add_fields(err_u_y, init_u_y, err_u_y, -1.)
    scale_field(err_u_x, 0.5, err_u_x)
    scale_field(err_u_y, 0.5, err_u_y)                
    advect_u(err_u_x, err_u_y, tmp_u_x, tmp_u_y,T_x, T_y,  psi_x, psi_y, dx) 
    add_fields(u_x, tmp_u_x, err_u_x, -1.)
    add_fields(u_y, tmp_u_y, err_u_y, -1.)
    if BFECC_clamp:
        clamp_u(err_u_x, u_x, tmp_u_x)
        clamp_u(err_u_y, u_y, tmp_u_y)
        copy_to(tmp_u_x, u_x)
        copy_to(tmp_u_y, u_y)
    else:
        copy_to(err_u_x, u_x)
        copy_to(err_u_y, u_y)

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
    init_u_x, init_u_y, u_x, u_y, 
    err_u_x, err_u_y, tmp_u_x, tmp_u_y, 
    T_x, T_y, psi_x, psi_y,
    F_x, F_y, phi_x, phi_y,
    dx, BFECC_clamp
):
    advect_u2(init_u_x, init_u_y, u_x, u_y, T_x, T_y, psi_x, psi_y, dx)
    advect_u2(u_x, u_y, err_u_x, err_u_y,F_x, F_y, phi_x, phi_y, dx)                
    add_fields(err_u_x, init_u_x, err_u_x, -1.) 
    add_fields(err_u_y, init_u_y, err_u_y, -1.)
    scale_field(err_u_x, 0.5, err_u_x)
    scale_field(err_u_y, 0.5, err_u_y)                
    advect_u2(err_u_x, err_u_y, tmp_u_x, tmp_u_y,T_x, T_y,  psi_x, psi_y, dx) 
    add_fields(u_x, tmp_u_x, err_u_x, -1.)
    add_fields(u_y, tmp_u_y, err_u_y, -1.)
    if BFECC_clamp:
        clamp_u(err_u_x, u_x, tmp_u_x)
        clamp_u(err_u_y, u_y, tmp_u_y)
        copy_to(tmp_u_x, u_x)
        copy_to(tmp_u_y, u_y)
    else:
        copy_to(err_u_x, u_x)
        copy_to(err_u_y, u_y)

@ti.kernel
def accumulate_init(
    f_x:ti.template(), f_y:ti.template(), init_u_x:ti.template(),init_u_y:ti.template(),
    F_x:ti.template(), F_y:ti.template(), phi_x:ti.template(), phi_y:ti.template(),
    dx:float, dt:float
):
    for I in ti.grouped(init_u_x):
        v, tem3,tem1,tem2= interp_u_MAC_grad(f_x, f_y,  phi_x[I], dx)
        init_u_x[I]+=F_x[I]@v*dt 

    for I in ti.grouped(init_u_y):
        v, tem3,tem1,tem2= interp_u_MAC_grad(f_x, f_y,  phi_y[I], dx)
        init_u_y[I]+=F_y[I]@v*dt 


@ti.kernel
def accumulate_init_new(
    f_x:ti.template(), f_y:ti.template(), init_u_x:ti.template(),init_u_y:ti.template(),
    F_x:ti.template(), F_y:ti.template(), phi_x:ti.template(), phi_y:ti.template(),
    H_x_prev:ti.template(), H_x_prev_prev:ti.template(),
    H_y_prev:ti.template(), H_y_prev_prev:ti.template(),
    ind:int,
    dx:float, dt:float
):
    for I in ti.grouped(init_u_x):
        v, tem3,tem1,tem2= interp_u_MAC_grad(f_x, f_y,  phi_x[I], dx)
        if(ind == 0):
            init_u_x[I]+=F_x[I]@v*dt 
        elif(ind == 1):
            init_u_x[I]+=(23*F_x[I]@v-16*F_x[I]@v+5*H_x_prev[I])/12*dt 
        else:
            init_u_x[I]+=(23*F_x[I]@v-16*H_x_prev[I]+5*H_x_prev_prev[I])/12*dt 
        H_x_prev_prev[I] = H_x_prev[I]
        H_x_prev[I] = F_x[I]@v


    for I in ti.grouped(init_u_y):
        v, tem3,tem1,tem2= interp_u_MAC_grad(f_x, f_y,  phi_y[I], dx)
        if(ind == 0):
            init_u_y[I]+=F_y[I]@v*dt 
        elif(ind == 1):
            init_u_y[I]+=(23*F_y[I]@v-16*F_y[I]@v+5*H_y_prev[I])/12*dt 
        else:
            init_u_y[I]+=(23*F_y[I]@v-16*H_y_prev[I]+5*H_y_prev_prev[I])/12*dt 
        H_y_prev_prev[I] = H_y_prev[I]
        H_y_prev[I] = F_y[I]@v
