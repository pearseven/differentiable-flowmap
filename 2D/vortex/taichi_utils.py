import taichi as ti

@ti.func
def merge_vec_8float(v1,v2):
    v = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for i in range(v1.n):
        v[i] = v1[i]
    for i in range(v2.n):
        v[v1.n+i] = v2[i]
    return v

@ti.func
def merge_vec_8int(v1,v2):
    v = ti.Vector([0,0,0,0,0,0,0,0])
    for i in range(v1.n):
        v[i] = v1[i]
    for i in range(v2.n):
        v[v1.n+i] = v2[i]
    return v

@ti.kernel
def scalar2vec(u_x:ti.template(), u_y:ti.template(), u:ti.template()):
    
    for i,j in u_x:
        u[i,j][0] = u_x[i,j]
    
    for i,j in u_y:
        u[i,j][1] = u_y[i,j]

@ti.kernel
def vec2scalar(u_x:ti.template(), u_y:ti.template(), u:ti.template()):
    
    for i,j in u_x:
        u_x[i,j] = u[i,j][0]
    
    for i,j in u_y:
        u_y[i,j] = u[i,j][1]

@ti.kernel
def vec2mat(F_x:ti.template(), F_y:ti.template(), F:ti.template()):
    
    for i,j in F_x:
        F[i,j][0,:] = F_x[i,j]
    
    for i,j in F_y:
        F[i,j][1,:] = F_y[i,j]

@ti.kernel
def mat2vec(F_x:ti.template(), F_y:ti.template(), F:ti.template()):
    for i,j in F_x:
        F_x[i,j] = F[i,j][0,:]
    
    for i,j in F_y:
        F_y[i,j] = F[i,j][1,:]


@ti.func
def valid(i,j, x):
    ii,jj = x.shape
    ind = True
    if(i<0 or j<0 or i>=ii or j>=jj):
        ind = False
    return ind

@ti.func
def flat(i,j, x):
    ii,jj = x.shape
    ind = i*jj+j
    if(i<0 or j<0 or i>=ii or j>=jj):
        ind = -1
    return int(ind)

@ti.func
def unflat(ind, x):
    ii,jj = x.shape
    i,j = int(ind/jj), ind%jj
    return i,j

eps = 1.e-6
data_type = ti.f32
@ti.func
def taylor_x(x, t, viscosity):
    return ti.sin(x[0]) * ti.cos(x[1]) * ti.exp(-2 * viscosity * t)
@ti.func
def taylor_y(x, t, viscosity):
    return -ti.cos(x[0]) * ti.sin(x[1]) * ti.exp(-2 * viscosity * t)

@ti.func
def hybrid_add(a,b):
    res= 0.0
    if(a==0):
        res = b
    else:
        res = (a+b)/2
    return res

@ti.func
def valid_center(i,j,v):
    ii,jj = v.shape
    res = False
    if(i>=0 and i<ii and j>=0 and j<jj):
        res= True 
    return res       

@ti.kernel
def scale_field(a: ti.template(), alpha: float, result: ti.template()):
    for i, j in result:
        result[i, j] = alpha * a[i,j]

@ti.kernel
def add_fields(f1: ti.template(), f2: ti.template(), dest: ti.template(), multiplier: float):
    for I in ti.grouped(dest):
        dest[I] = f1[I] + multiplier * f2[I]

@ti.kernel
def calc_square(f: ti.template(),  dest: ti.template()):
    for I in ti.grouped(dest):
        dest[I] = (f[I][0]*f[I][0]+f[I][1]*f[I][1])**0.5

@ti.kernel
def copy_to(source: ti.template(), dest: ti.template()):
    for I in ti.grouped(source):
        dest[I] = source[I]

@ti.func
def sample(qf: ti.template(), u: float, v: float):
    u_dim, v_dim = qf.shape
    i = ti.max(0, ti.min(int(u), u_dim-1))
    j = ti.max(0, ti.min(int(v), v_dim-1))
    return qf[i, j]

@ti.func
def lerp(vl, vr, frac):
    # frac: [0.0, 1.0]
    return vl + frac * (vr - vl)

@ti.kernel
def center_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i + 0.5, j + 0.5]) * dx

@ti.kernel
def horizontal_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i, j + 0.5]) * dx

@ti.kernel
def vertical_coords_func(pf: ti.template(), dx: float):
    for i, j in pf:
        pf[i, j] = ti.Vector([i + 0.5, j]) * dx

@ti.kernel
def get_central_vector(horizontal: ti.template(), vertical: ti.template(), central: ti.template()):
    for i, j in central:
        central[i,j].x = 0.5 * (horizontal[i+1, j] + horizontal[i, j])
        central[i,j].y = 0.5 * (vertical[i, j+1] + vertical[i, j])

@ti.kernel
def split_central_vector(central: ti.template(), horizontal: ti.template(), vertical: ti.template()):
    for i, j in horizontal:
        r = sample(central, i, j)
        l = sample(central, i-1, j)
        horizontal[i,j] = 0.5 * (r.x + l.x)
    for i, j in vertical:
        t = sample(central, i, j)
        b = sample(central, i, j-1)
        vertical[i,j] = 0.5 * (t.y + b.y)

@ti.kernel
def sizing_function(u: ti.template(), sizing: ti.template(), dx: float):
    u_dim, v_dim = u.shape
    for i, j in u:
        u_l = sample(u, i-1, j)
        u_r = sample(u, i+1, j)
        u_t = sample(u, i, j+1)
        u_b = sample(u, i, j-1)
        partial_x = 1./(2*dx) * (u_r - u_l)
        partial_y = 1./(2*dx) * (u_t - u_b)
        if i == 0:
            partial_x = 1./(2*dx) * (u_r - 0)
        elif i == u_dim - 1:
            partial_x = 1./(2*dx) * (0 - u_l)
        if j == 0:
            partial_y = 1./(2*dx) * (u_t - 0)
        elif j == v_dim - 1:
            partial_y = 1./(2*dx) * (0 - u_b)

        sizing[i, j] = ti.sqrt(partial_x.x ** 2 + partial_x.y ** 2 + partial_y.x ** 2 + partial_y.y ** 2)

@ti.kernel
def diffuse_grid(value: ti.template(), tmp: ti.template()):
    for i, j in tmp:
        tmp[i,j] = 0.25 * (sample(value, i+1, j) + sample(value, i-1, j)\
                + sample(value, i, j+1) + sample(value, i, j-1))
    for i, j in tmp:
        value[i,j] = ti.max(value[i,j], tmp[i,j])

@ti.kernel
def curl(vf: ti.template(), cf: ti.template(), dx: float):
    for i, j in vf:
        vl = sample(vf, i - 1, j)
        vr = sample(vf, i + 1, j)
        vb = sample(vf, i, j - 1)
        vt = sample(vf, i, j + 1)
        cf[i, j] = (vr.y - vl.y - vt.x + vb.x) / (2*dx)

# # # # interpolation kernels # # # # 

@ti.func
def N_1(x):
    return 1.0-ti.abs(x)
    
@ti.func
def dN_1(x):
    result = -1.0
    if x < 0.:
        result = 1.0
    return result

@ti.func
def interp_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0., ti.min(u, u_dim-1-eps))
    t = ti.max(0., ti.min(v, v_dim-1-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)

    interped = a * N_1(fu) * N_1(fv) + \
            b * N_1(fu-1) * N_1(fv) + \
            c * N_1(fu) * N_1(fv-1) + \
            d * N_1(fu-1) * N_1(fv-1)
    
    return interped

@ti.func
def interp_grad_1(vf, p, dx, BL_x = 0.5, BL_y = 0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0., ti.min(u, u_dim-1-eps))
    t = ti.max(0., ti.min(v, v_dim-1-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)

    # comp grad while bilerp
    partial_x = 1./dx * (a * dN_1(fu) * N_1(fv) + \
                    b * dN_1(fu-1) * N_1(fv) + \
                    c * dN_1(fu) * N_1(fv-1) + \
                    d * dN_1(fu-1) * N_1(fv-1))
    partial_y = 1./dx * (a * N_1(fu) * dN_1(fv) + \
                        b * N_1(fu-1) * dN_1(fv) + \
                        c * N_1(fu) * dN_1(fv-1) + \
                        d * N_1(fu-1) * dN_1(fv-1))

    interped = a * N_1(fu) * N_1(fv) + \
            b * N_1(fu-1) * N_1(fv) + \
            c * N_1(fu) * N_1(fv-1) + \
            d * N_1(fu-1) * N_1(fv-1)
    
    return interped, ti.Vector([partial_x, partial_y])

@ti.func
def interp_1_with_grad(vf, p, dx, BL_x = 0.5, BL_y = 0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(0., ti.min(u, u_dim-1-eps))
    t = ti.max(0., ti.min(v, v_dim-1-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)
    # fract
    fu, fv = s - iu, t - iv
    a = sample(vf, iu, iv)
    b = sample(vf, iu + 1, iv)
    c = sample(vf, iu, iv + 1)
    d = sample(vf, iu + 1, iv + 1)

    # comp grad while bilerp
    interped = a * N_1(fu) * N_1(fv) + \
        b * N_1(fu-1) * N_1(fv) + \
        c * N_1(fu) * N_1(fv-1) + \
        d * N_1(fu-1) * N_1(fv-1)
    dres_dvel_id = ti.Vector([0,0,0,0])
    dres_dvel_val = ti.Vector([0.0,0.0,0.0,0.0])
    dres_dvel_id[0],dres_dvel_id[1],dres_dvel_id[2],dres_dvel_id[3] =    flat(iu + 1, iv + 1,vf),       flat(iu + 1, iv,vf),        flat(iu, iv + 1,vf),        flat(iu, iv,vf)
    dres_dvel_val[0],dres_dvel_val[1],dres_dvel_val[2],dres_dvel_val[3] = N_1(fu-1) * N_1(fv-1),        N_1(fu-1) * N_1(fv),        N_1(fu) * N_1(fv-1),        N_1(fu) * N_1(fv)

    partial_x = 1./dx * (a * dN_1(fu) * N_1(fv) + \
                    b * dN_1(fu-1) * N_1(fv) + \
                    c * dN_1(fu) * N_1(fv-1) + \
                    d * dN_1(fu-1) * N_1(fv-1))
    partial_y = 1./dx * (a * N_1(fu) * dN_1(fv) + \
                        b * N_1(fu-1) * dN_1(fv) + \
                        c * N_1(fu) * dN_1(fv-1) + \
                        d * N_1(fu-1) * dN_1(fv-1))
    dres_dpos = ti.Vector([partial_x, partial_y])
    
    return interped, dres_dpos, dres_dvel_val,dres_dvel_id

# https://www.sciencedirect.com/science/article/pii/S0021999103006296
@ti.func
def N_4(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 115.0/192 - (5 * abs_x**2) / 8 + abs_x**4 / 4
    elif abs_x < 1.5:
        result = 55.0/96 + (5 * abs_x) / 24 - 5*abs_x**2/4 + (5 * abs_x**3) / 6 - abs_x**4 / 6
    elif abs_x < 2.5:
        result = 625.0/384 - (125 * abs_x) / 48 + (25 * abs_x**2) / 16 - (5 * abs_x**3) / 12 + abs_x**4 / 24
    else:
        result = 0
    return result
    
    

@ti.func
def dN_4(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = (-5.0/4 * abs_x + abs_x**3) 
    elif abs_x < 1.5:
        result = (5.0/24 - 5.0/2 * abs_x + 5.0/2 * abs_x**2 - 2.0/3 * abs_x**3) 
    elif abs_x < 2.5:
        result = (-125.0/48 + 25.0/8 * abs_x - 5.0/4 * abs_x**2 + 1.0/6 * abs_x**3) 

    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def N_3(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = 1.0/2*abs_x**3 - abs_x**2+2.0/3
    elif abs_x < 2.0:
        result = 1.0/6*(2-abs_x)**3
    return result

    
@ti.func
def dN_3(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 1.0:
        result = 3.0/2*abs_x**2 - 2*abs_x
    elif abs_x < 2.0:
        result = -1.0/2*(2-abs_x)**2

    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result


@ti.func
def N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = 3.0/4.0 - abs_x ** 2
    elif abs_x < 1.5:
        result = 0.5 * (3.0/2.0-abs_x) ** 2
    return result
    
@ti.func
def dN_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2 * abs_x
    elif abs_x < 1.5:
        result = 0.5 * (2 * abs_x - 3)
    if x < 0.0: # if x < 0 then abs_x is -1 * x
        result *= -1
    return result

@ti.func
def d2N_2(x):
    result = 0.0
    abs_x = ti.abs(x)
    if abs_x < 0.5:
        result = -2
    elif abs_x < 1.5:
        result = 1
    return result


@ti.func
def interp_grad_2(vf, p, dx, BL_x=0.5, BL_y=0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            partial_x += 1. / dx * (value * dN_2(x_p_x_i) * N_2(y_p_y_i))
            partial_y += 1. / dx * (value * N_2(x_p_x_i) * dN_2(y_p_y_i))
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            vector_value = ti.Vector([value, 0.0])
            if is_y:
                vector_value = ti.Vector([0.0, value])
            new_C += 4 * N_2(x_p_x_i) * N_2(y_p_y_i) * vector_value.outer_product(dpos) / dx

    return interped, ti.Vector([partial_x, partial_y]), new_C


@ti.func
def interp_grad_3(vf, p, dx, BL_x=0.5, BL_y=0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-2, 4):
        for j in range(-2, 4):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            partial_x += 1. / dx * (value * dN_3(x_p_x_i) * N_3(y_p_y_i))
            partial_y += 1. / dx * (value * N_3(x_p_x_i) * dN_3(y_p_y_i))
            interped += value * N_3(x_p_x_i) * N_3(y_p_y_i)
            dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            vector_value = ti.Vector([value, 0.0])
            if is_y:
                vector_value = ti.Vector([0.0, value])
            new_C += 4 * N_3(x_p_x_i) * N_3(y_p_y_i) * vector_value.outer_product(dpos) / dx

    return interped, ti.Vector([partial_x, partial_y]), new_C

@ti.func
def interp_grad_4(vf, p, dx, BL_x=0.5, BL_y=0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-3, 5):
        for j in range(-3, 5):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            partial_x += 1. / dx * (value * dN_4(x_p_x_i) * N_4(y_p_y_i))
            partial_y += 1. / dx * (value * N_4(x_p_x_i) * dN_4(y_p_y_i))
            interped += value * N_4(x_p_x_i) * N_4(y_p_y_i)
            dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            vector_value = ti.Vector([value, 0.0])
            if is_y:
                vector_value = ti.Vector([0.0, value])
            new_C += 4 * N_4(x_p_x_i) * N_4(y_p_y_i) * vector_value.outer_product(dpos) / dx

    return interped, ti.Vector([partial_x, partial_y]), new_C

@ti.func
def interp_u_MAC_vector(u_x, u_y, p, dx):
    u_x_p = interp_2_vector(u_x, p, dx, BL_x = 0.0, BL_y = 0.5)
    u_y_p = interp_2_vector(u_y, p, dx, BL_x = 0.5, BL_y = 0.0)
    return u_x_p, u_y_p

@ti.func
def interp_2_vector(vf, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    interped = ti.Vector([0.0,0.0])

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped

@ti.func
def interp_2_scalar(vf, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    interped = 0.0

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped

# @ti.func
# def interp_center(vf, p, dx, BL_x=0.5, BL_y=0.5):
#     u_dim, v_dim = vf.shape

#     u, v = p / dx
#     u = u - BL_x
#     v = v - BL_y
#     s = ti.max(1., ti.min(u, u_dim - 2 - eps))
#     t = ti.max(1., ti.min(v, v_dim - 2 - eps))

#     # floor
#     iu, iv = ti.floor(s), ti.floor(t)
#     interped = 0.

#     # loop over 16 indices
#     for i in range(-1, 3):
#         for j in range(-1, 3):
#             x_p_x_i = s - (iu + i)
#             y_p_y_i = t - (iv + j)
#             value = sample(vf, iu + i, iv + j)
#             interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
#     return interped


@ti.func
def divergence_2(T, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = T.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    grad_T = 0.0

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            T_value = sample(T, iu + i, iv + j)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            dw = ti.Vector([dw_x, dw_y])
            grad_T += T_value @ dw

    return grad_T

@ti.func
def interp_MAC_divergence(T_x, T_y, p, dx):
    grad_T_x = divergence_2(T_x, p, dx, BL_x=0.0, BL_y=0.5)
    grad_T_y = divergence_2(T_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([grad_T_x, grad_T_y])

@ti.func
def interp_u_MAC_grad(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_2(u_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y = interp_grad_2(u_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y

@ti.func
def interp_u_MAC_grad3(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_4(u_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y = interp_grad_4(u_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y

@ti.func
def interp_u_MAC_1(u_x, u_y, p, dx):
    u_x_p = interp_1(u_x, p, dx, BL_x=0.0, BL_y=0.5)
    u_y_p = interp_1(u_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([u_x_p, u_y_p])

@ti.func
def interp_u_MAC_1_with_grad(u_x, u_y, p, dx):
    u_x_p,du_x_res_dpos, d_res_dvel_val_x,d_res_dvel_id_x = interp_1_with_grad(u_x, p, dx, BL_x=0.0, BL_y=0.5)
    u_y_p,du_y_res_dpos, d_res_dvel_val_y,d_res_dvel_id_y = interp_1_with_grad(u_y, p, dx, BL_x=0.5, BL_y=0.0)
    return ti.Vector([u_x_p, u_y_p]),ti.Matrix.rows([du_x_res_dpos, du_y_res_dpos]),d_res_dvel_val_x,d_res_dvel_id_x, d_res_dvel_val_y, d_res_dvel_id_y

@ti.func
def interp_grad_2_imp(vf, p, dx, BL_x = 0.5, BL_y = 0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim-2-eps))
    t = ti.max(1., ti.min(v, v_dim-2-eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0])
    # interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            # imp_value = sample(imp, iu + i, iv + j)
            dw_x = 1./dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1./dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            partial_x += value * dw_x
            partial_y += value * dw_y
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            # dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            # vector_value = ti.Vector([imp_value, 0.0])
            # # vector_value = ti.Vector([value, 0.0])
            # if is_y:
            #     vector_value = ti.Vector([0.0, imp_value])
            #     # vector_value = ti.Vector([0.0, value])
            new_C += ti.Vector([dw_x, dw_y]) * value
            # interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)
    
    return interped, ti.Vector([partial_x, partial_y]), new_C

@ti.func
def interp_u_MAC_grad_imp(u_x, u_y, p, dx):
    u_x_p, grad_u_x_p, C_x = interp_grad_2_imp(u_x, p, dx, BL_x = 0.0, BL_y = 0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y = interp_grad_2_imp(u_y, p, dx, BL_x = 0.5, BL_y = 0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y


@ti.func
def interp_grad_2_imp_and_grad_imp(vf, imp, p, dx, BL_x=0.5, BL_y=0.5, is_y=False):
    u_dim, v_dim = vf.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    partial_x = 0.
    partial_y = 0.
    interped = 0.

    new_C = ti.Vector([0.0, 0.0])
    interped_imp = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            value = sample(vf, iu + i, iv + j)
            imp_value = sample(imp, iu + i, iv + j)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            partial_x += value * dw_x
            partial_y += value * dw_y
            interped += value * N_2(x_p_x_i) * N_2(y_p_y_i)
            # dpos = ti.Vector([-x_p_x_i, -y_p_y_i])
            # vector_value = ti.Vector([imp_value, 0.0])
            # # vector_value = ti.Vector([value, 0.0])
            # if is_y:
            #     vector_value = ti.Vector([0.0, imp_value])
            #     # vector_value = ti.Vector([0.0, value])
            new_C += ti.Vector([dw_x, dw_y]) * imp_value
            interped_imp += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped, ti.Vector([partial_x, partial_y]), new_C, interped_imp


@ti.func
def interp_u_MAC_imp_and_grad_imp(u_x, u_y, imp_x, imp_y, p, dx):
    u_x_p, grad_u_x_p, C_x, interped_imp_x = interp_grad_2_imp_and_grad_imp(u_x, imp_x, p, dx, BL_x=0.0, BL_y=0.5, is_y=False)
    u_y_p, grad_u_y_p, C_y, interped_imp_y = interp_grad_2_imp_and_grad_imp(u_y, imp_y, p, dx, BL_x=0.5, BL_y=0.0, is_y=True)
    return ti.Vector([u_x_p, u_y_p]), ti.Matrix.rows([grad_u_x_p, grad_u_y_p]), C_x, C_y, interped_imp_x, interped_imp_y


@ti.func
def interp(field, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = field.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    interped = 0.

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            imp_value = sample(field, iu + i, iv + j)
            interped += imp_value * N_2(x_p_x_i) * N_2(y_p_y_i)

    return interped


@ti.func
def interp_grad_2_updated_imp(imp, phi_grid, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = imp.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    new_C = ti.Vector([0.0, 0.0])

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            updated_imp_pos = sample(phi_grid, iu + i, iv + j)
            imp_value = interp(imp, updated_imp_pos, dx, BL_x, BL_y)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            new_C += ti.Vector([dw_x, dw_y]) * imp_value

    return new_C


@ti.func
def interp_u_MAC_grad_updated_imp(u_x, u_y, imp_x, imp_y, phi_x_grid, phi_y_grid, p, dx):
    C_x = interp_grad_2_updated_imp(imp_x, phi_x_grid, p, dx, BL_x=0.0, BL_y=0.5)
    C_y = interp_grad_2_updated_imp(imp_y, phi_y_grid, p, dx, BL_x=0.5, BL_y=0.0)
    return C_x, C_y

@ti.func
def interp_grad_T(T, p, dx, BL_x=0.5, BL_y=0.5):
    u_dim, v_dim = T.shape

    u, v = p / dx
    u = u - BL_x
    v = v - BL_y
    s = ti.max(1., ti.min(u, u_dim - 2 - eps))
    t = ti.max(1., ti.min(v, v_dim - 2 - eps))

    # floor
    iu, iv = ti.floor(s), ti.floor(t)

    grad_T = ti.Matrix.zero(float, 2, 2)

    # loop over 16 indices
    for i in range(-1, 3):
        for j in range(-1, 3):
            x_p_x_i = s - (iu + i)
            y_p_y_i = t - (iv + j)
            T_value = sample(T, iu + i, iv + j)
            dw_x = 1. / dx * dN_2(x_p_x_i) * N_2(y_p_y_i)
            dw_y = 1. / dx * N_2(x_p_x_i) * dN_2(y_p_y_i)
            dw = ti.Vector([dw_x, dw_y])
            grad_T += dw.outer_product(T_value)

    return grad_T

@ti.func
def interp_MAC_grad_T(T_x, T_y, p, dx):
    grad_T_x = interp_grad_T(T_x, p, dx, BL_x=0.0, BL_y=0.5)
    grad_T_y = interp_grad_T(T_y, p, dx, BL_x=0.5, BL_y=0.0)
    return grad_T_x, grad_T_y

# # # # ti and torch conversion # # # # 

@ti.kernel
def random_initialize(data: ti.types.ndarray()):
    for I in ti.grouped(data):
        data[I] = (ti.random() * 2.0 - 1.0) * 1e-4

    
    
def diffuse_field(field_temp, field, coe):
    copy_to(field, field_temp)
    for it in range(20):
        GS(field, field_temp, coe)
    copy_to(field_temp, field)

@ti.kernel
def GS(field:ti.template(), field_temp:ti.template(), coe:float):
    for i, j in field_temp:
        if (i + j)%2==0:
            field_temp[i, j] = (field[i, j] + coe * (
                                sample(field_temp, i - 1, j) +
                                sample(field_temp, i + 1, j) +
                                sample(field_temp, i, j - 1) +
                                sample(field_temp, i, j + 1)
                        )) / (1.0 + 4.0 * coe)
    for i, j in field_temp:
        if (i + j)%2==1:
            field_temp[i, j] = (field[i, j] + coe * (
                                sample(field_temp, i - 1, j) +
                                sample(field_temp, i + 1, j) +
                                sample(field_temp, i, j - 1) +
                                sample(field_temp, i, j + 1)
                        )) / (1.0 + 4.0 * coe)

@ti.kernel
def np2taichi(np_data:ti.types.ndarray(dtype=ti.f32, ndim=3), taichi_data:ti.template()):
    for I in ti.grouped(taichi_data):
        taichi_data[I][0] = np_data[I, 0]
        taichi_data[I][1] = np_data[I, 1]

"""
@ti.kernel
def sparse2dense(sparse:ti.template(),dense:ti.template()):
    for I in ti.grouped(dense):
        dense[I]=sparse[I]

@ti.kernel
def sparse2dense_2D_split(sparse:ti.template(),dense:ti.template(),ind:int):
    for I in ti.grouped(dense):
        dense[I]=sparse[ind,flat(I[0],I[1], dense)]


def dense2sparse(dense:ti.template(),K:ti.types.sparse_matrix_builder()):
    dense2sparse_kernel(dense,K)
    return K.build()

@ti.kernel
def dense2sparse_kernel(dense:ti.template(),K:ti.types.sparse_matrix_builder()):
    for I in ti.grouped(dense):
        K[I] = dense[I]
"""