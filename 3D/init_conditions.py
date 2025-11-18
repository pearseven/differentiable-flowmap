# 
from taichi_utils import *
import math
from hyperparameters import *
import numpy as np
import random
# w: vortex strength
# rad: radius of torus
# delta: thickness of torus
# c: ring center position
# unit_x, unit_y: the plane of the circle



@ti.kernel
def segment_sample(w:ti.template(),w_num:ti.template(),w_c:ti.template(),w_w:ti.template(),w_r:ti.template(), w_theshhold:float):
    w_num[None] = 0
    for i,j,k in w:
        #p = ti.Vector([i+0.5,j+0.5,k+0.5])*dx
        p = ti.Vector([i+1,j+1,k+1])*dx
        if(w[i,j,k].norm()>w_theshhold and i%2==0 and j%2==0 and k%2==0):
            ind = ti.atomic_add(w_num[None],1)
            w_c[ind] = p
            w_w[ind] = w[i,j,k]*dx*dx*dx*2*2*2
            w_r[ind] = 0.016#0.016 #0.01



@ti.func
def segment2v(
    x_p,
    x_n,
    strength,
    delta,
    p
):
    cross = (x_n - p).cross(x_p - p)
    res =strength * (
        (x_p - p)/((x_p - p).norm()+1e-5)*(1-ti.exp(-(x_p - p).norm()/delta))- 
        (x_n - p)/((x_n - p).norm()+1e-5)*(1-ti.exp(-(x_n - p).norm()/delta))
    )*(x_p-x_n).norm()* cross/(cross.norm()**2+1e-5) * (1-ti.exp(-(cross.norm()/delta)**2))
    return res


def segment_velocity_gradient(
    segment_num:int,
    delta:ti.template(),
    strength:ti.template(),
    x_p:ti.template(),
    x_n:ti.template(),
    adj_delta:ti.template(),
    adj_strength:ti.template(),
    adj_x_p:ti.template(),
    adj_x_n:ti.template(),
    pf: ti.template(),
    adj_vf:ti.template(),
):
    for i in range(segment_num):
        segment_velocity_gradient_kernel(
            delta, strength, x_p, x_n, adj_delta, adj_strength, adj_x_p, adj_x_n, pf, adj_vf, i,
        )    
    adj_delta_np = adj_delta.to_numpy()
    adj_strength_np = adj_strength.to_numpy()
    adj_x_p_np = adj_x_p.to_numpy()
    adj_x_n_np = adj_x_n.to_numpy()

    # 2. 检查是否有 NaN
    def check_nan(array, name):
        if np.isnan(array).any():
            print(f"⚠️ 检查到 {name} 中有 NaN！")
        else:
            print(f"✅ {name} 中没有 NaN。")

    check_nan(adj_delta_np, "adj_delta")
    check_nan(adj_strength_np, "adj_strength")
    check_nan(adj_x_p_np, "adj_x_p")
    check_nan(adj_x_n_np, "adj_x_n")

@ti.kernel
def segment_velocity_gradient_kernel(
    delta:ti.template(),
    strength:ti.template(),
    x_p:ti.template(),
    x_n:ti.template(),
    adj_delta:ti.template(),
    adj_strength:ti.template(),
    adj_x_p:ti.template(),
    adj_x_n:ti.template(),
    pf: ti.template(),
    adj_vf:ti.template(),
    ind:int,
):
    adj_delta[ind] = 0.0
    adj_strength[ind] = 0.0
    adj_x_p[ind] = ti.Vector([0.0,0.0,0.0])
    adj_x_n[ind] = ti.Vector([0.0,0.0,0.0])
    epsilon = 1e-4
    for i,j,k in adj_vf:
        p = pf[i,j,k]
        a = segment2v(x_p[ind], x_n[ind], strength[ind], delta[ind], p)
        a1 = segment2v(x_p[ind]+ti.Vector([epsilon,0,0]), x_n[ind], strength[ind], delta[ind], p)
        a2 = segment2v(x_p[ind]+ti.Vector([0,epsilon,0]), x_n[ind], strength[ind], delta[ind], p)
        a3 = segment2v(x_p[ind]+ti.Vector([0,0,epsilon]), x_n[ind], strength[ind], delta[ind], p)
        a4 = segment2v(x_p[ind], x_n[ind]+ti.Vector([epsilon,0,0]), strength[ind], delta[ind], p)
        a5 = segment2v(x_p[ind], x_n[ind]+ti.Vector([0,epsilon,0]), strength[ind], delta[ind], p)
        a6 = segment2v(x_p[ind], x_n[ind]+ti.Vector([0,0,epsilon]), strength[ind], delta[ind], p)
        a7 = segment2v(x_p[ind], x_n[ind], strength[ind]+epsilon, delta[ind], p)
        a8 = segment2v(x_p[ind], x_n[ind], strength[ind], delta[ind]+epsilon, p)
        adj_a1 = adj_vf[i,j,k].dot((a1-a)/epsilon)
        adj_a2 = adj_vf[i,j,k].dot((a2-a)/epsilon)
        adj_a3 = adj_vf[i,j,k].dot((a3-a)/epsilon)
        adj_a4 = adj_vf[i,j,k].dot((a4-a)/epsilon)
        adj_a5 = adj_vf[i,j,k].dot((a5-a)/epsilon)
        adj_a6 = adj_vf[i,j,k].dot((a6-a)/epsilon)
        adj_a7 = adj_vf[i,j,k].dot((a7-a)/epsilon)
        adj_a8 = adj_vf[i,j,k].dot((a8-a)/epsilon)
        adj_delta[ind]+=adj_a8
        adj_strength[ind]+=adj_a7
        adj_x_p[ind]+=ti.Vector([adj_a1,adj_a2,adj_a3])
        adj_x_n[ind]+=ti.Vector([adj_a4,adj_a5,adj_a6])



@ti.func
def segment2v_origin(x_p, x_n, delta,p):
    p_diff = p-x_p
    r = p_diff.norm()
    return (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(x_n)

@ti.kernel
def segment_velocity_origin(
    segment_num:int,
    x_p:ti.template(),
    x_n:ti.template(),
    delta:ti.template(), 
    pf: ti.template(),
    vf:ti.template()
):
    vf.fill(0.0)
    for i, j, k in vf:
        p = pf[i,j,k]
        for l in range(segment_num):
            if((x_p[l] - p).norm()< 30*dx):
                vf[i,j,k] += segment2v_origin(x_p[l], x_n[l], delta[l],pf[i,j,k])

@ti.kernel
def segment_velocity_origin_gradient(
    segment_num:int,
    delta:ti.template(),
    x_p:ti.template(),
    x_n:ti.template(),
    adj_delta:ti.template(),
    adj_x_p:ti.template(),
    adj_x_n:ti.template(),
    pf: ti.template(),
    adj_vf:ti.template(),
):
    adj_delta.fill(0.0)
    adj_x_p.fill(0.0)
    adj_x_n.fill(0.0)
    epsilon = 1e-4
    for i,j,k in adj_vf:
        p = pf[i,j,k]
        for ind in range(segment_num):
            if((x_p[ind] - p).norm()< 30*dx):
                a = segment2v_origin(x_p[ind], x_n[ind], delta[ind], p)
                a1 = segment2v_origin(x_p[ind]+ti.Vector([epsilon,0,0]), x_n[ind], delta[ind], p)
                a2 = segment2v_origin(x_p[ind]+ti.Vector([0,epsilon,0]), x_n[ind], delta[ind], p)
                a3 = segment2v_origin(x_p[ind]+ti.Vector([0,0,epsilon]), x_n[ind], delta[ind], p)
                a4 = segment2v_origin(x_p[ind], x_n[ind]+ti.Vector([epsilon,0,0]), delta[ind], p)
                a5 = segment2v_origin(x_p[ind], x_n[ind]+ti.Vector([0,epsilon,0]), delta[ind], p)
                a6 = segment2v_origin(x_p[ind], x_n[ind]+ti.Vector([0,0,epsilon]), delta[ind], p)
                a8 = segment2v_origin(x_p[ind], x_n[ind], delta[ind]+epsilon, p)
                adj_a1 = adj_vf[i,j,k].dot((a1-a)/epsilon)
                adj_a2 = adj_vf[i,j,k].dot((a2-a)/epsilon)
                adj_a3 = adj_vf[i,j,k].dot((a3-a)/epsilon)
                adj_a4 = adj_vf[i,j,k].dot((a4-a)/epsilon)
                adj_a5 = adj_vf[i,j,k].dot((a5-a)/epsilon)
                adj_a6 = adj_vf[i,j,k].dot((a6-a)/epsilon)
                adj_a8 = adj_vf[i,j,k].dot((a8-a)/epsilon)
                adj_delta[ind]+=adj_a8
                adj_x_p[ind]+=ti.Vector([adj_a1,adj_a2,adj_a3])
                adj_x_n[ind]+=ti.Vector([adj_a4,adj_a5,adj_a6])




@ti.kernel
def segment_velocity(
    segment_num:int,

    delta:ti.template(),
    strength:ti.template(),
    x_p:ti.template(),
    x_n:ti.template(),
 
    pf: ti.template(),
    vf:ti.template()
):

    vf.fill(0.0)
    for i,j,k in vf:
        p = pf[i,j,k]
        for ii in range(segment_num):            
            #cross = (x_n[ii] - p).cross(x_p[ii] - p)
            vf[i,j,k] +=segment2v(
                            x_p[ii],
                            x_n[ii],
                            strength[ii],
                            delta[ii],
                            p
                        )    
            #strength[ii] * (
            #    (x_p[ii] - p)/(x_p[ii] - p).norm()*(1-ti.exp(-(x_p[ii] - p).norm()/delta[ii]))- 
            #    (x_n[ii] - p)/(x_n[ii] - p).norm()*(1-ti.exp(-(x_n[ii] - p).norm()/delta[ii]))
            #)*(x_p[ii]-x_n[ii]).norm()* cross/cross.norm()**2 * (1-ti.exp(-(cross.norm()/delta[ii])**2))

@ti.kernel
def set_simple_smoke(X:ti.template(),passive:ti.template()):
    for i, j, k in passive:
        p = X[i,j,k]
        #p2D = ti.Vector([p[1], p[2]])
        #passive[i,j,k] =  (0.75 - (p2D - ti.Vector([0.5,0.5])).norm())/0.75
        if(i>64):
            passive[i,j,k] = 0
        else:
            passive[i,j,k] = (64-i)/64* (1+(1-abs(p[1] - 0.5)/0.5))/2* (1+(1-abs(p[2] - 0.5)/0.5))/2


@ti.kernel
def set_leapfrog_smoke(X:ti.template(),w:ti.template(),passive:ti.template()):
    for i, j, k in passive:
        p = X[i,j,k]
        p2D = ti.Vector([p[1], p[2]])
        #r = (p2D - ti.Vector([0.5,0.5])).norm() 
        #r =abs(r- 0.21)
        #r_max = 0.2
        x_max = (0.16 + 0.5) + 0.625* 0.8
        x_min = 0.5
        if(p[0]< x_min):
            passive[i,j,k] = 1.0
        elif(p[0]>x_max):
            passive[i,j,k] = 0.0
        else:
            passive[i,j,k] = (x_max - p[0])/(x_max-x_min)
                
        #w_min = 0.1
        #w_max = 1.0
        #if(w[i,j,k].norm()<w_min):
        #    passive[i,j,k] =  0.0
        #elif(w[i,j,k].norm()>w_max):
        #    passive[i,j,k] =  1.0
        #else:
        #    passive[i,j,k] =  (w[i,j,k].norm()-w_min)/(w_max-w_min)

@ti.kernel
def set_plume_smoke(X:ti.template(),w:ti.template(),passive:ti.template()):
    for i, j, k in passive:
        p = X[i,j,k]
        p2D = ti.Vector([p[1], p[2]])
        if((p2D-ti.Vector([0.5,0.5])).norm()<0.3 and p[0]<0.7):
            passive[i,j,k] = (0.3-(p2D-ti.Vector([0.5,0.5])).norm())/0.3 * (0.7-p[0])/0.7
        else:
            passive[i,j,k] =  0

@ti.kernel
def set_boundary_mask(boundary_mask:ti.template(),boundary_vel:ti.template()):
    boundary_mask.fill(0.0)
    boundary_vel.fill(0.0)

@ti.kernel
def add_vortex_ring_and_smoke(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), num_samples: int):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point has an associated length
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector)
            smokef[i,j,k] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k] > 0.002:
            smokef[i,j,k] = 1.0
        else:
            smokef[i,j,k] = 0.0

@ti.kernel
def add_vortex_ring_and_smoke_with_coef(w: float, rad: float, delta: float, c: ti.types.vector(3, float),
                unit_x: ti.types.vector(3, float), unit_y: ti.types.vector(3, float),
                pf: ti.template(), vf: ti.template(), smokef: ti.template(), color: ti.types.vector(3, float), 
                num_samples: int, strength:float):
    curve_length = (2 * math.pi * rad) / num_samples # each sample point has an associated length
    for i, j, k in vf:
        for l in range(num_samples):
            theta = l/num_samples * 2 * (math.pi)
            p_sampled = rad * (ti.cos(theta) * unit_x + ti.sin(theta) * unit_y) + c # position of the sample point
            p_diff = pf[i,j,k]-p_sampled
            r = p_diff.norm()
            w_vector = w * (-ti.sin(theta) * unit_x + ti.cos(theta) * unit_y)
            vf[i,j,k] += curve_length * (-1/(4 * math.pi * r ** 3) * (1-ti.exp(-(r/delta) ** 3))) * p_diff.cross(w_vector) * strength
            smokef[i,j,k] += curve_length * (ti.exp(-(r/delta) ** 3))
    for i, j, k in smokef:
        if smokef[i,j,k] > 0.002:
            smokef[i,j,k] = 1.0
        else:
            smokef[i,j,k] = 0.0
    
def init_vorts_leapfrog(X, u, smoke1, smoke2):
    u.fill(0.0)

    smoke1.fill(0.)
    smoke2.fill(0.)
    # front and back revised
    radius = 0.21
    x_gap = 0.625 * radius
    x_start = (0.16 + 0.5)
    delta = 0.08 * radius
    w = radius * 0.1

    add_vortex_ring_and_smoke(w = w, rad = radius, delta = delta, c = ti.Vector([x_start,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke1, color = ti.Vector([1., 0, 0]), num_samples = 2000)

    add_vortex_ring_and_smoke(w = w, rad = radius, delta = delta, c = ti.Vector([(x_start) +x_gap,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 0, 1.]), num_samples = 2000)

    add_fields(smoke1, smoke2, smoke1, 1.0)

def init_one_vort(X, u, smoke1, smoke2):
    u.fill(0.0)

    smoke1.fill(0.)
    smoke2.fill(0.)

    radius = 0.21
    x_gap = 0.625 * radius
    x_start = (0.16 + 0.5)
    delta = 0.08 * radius
    w = radius * 0.1

    add_vortex_ring_and_smoke(w = w, rad = radius, delta = delta, c = ti.Vector([(x_start) +x_gap,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 0, 1.]), num_samples = 2000)

    add_fields(smoke1, smoke2, smoke1, 1.0)

def init_one_vort_with_coef(X, u, smoke1, smoke2, strength):
    smoke1.fill(0.)
    smoke2.fill(0.)
    u.fill(0.0)

    radius = 0.21
    x_gap = 0.625 * radius
    x_start = (0.16 + 0.5)
    delta = 0.2 * radius
    w = radius * 0.1

    add_vortex_ring_and_smoke_with_coef(w = w, rad = radius, delta = delta, c = ti.Vector([(x_start) +x_gap,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 0, 1.]), num_samples = 2000, strength= strength)

    add_fields(smoke1, smoke2, smoke1, 1.0)

def init_vorts_leapfrog_with_coef(X, u, smoke1, smoke2,strength):
    u.fill(0.0)

    smoke1.fill(0.)
    smoke2.fill(0.)
    # front and back revised
    radius = 0.21
    x_gap = 0.625 * radius
    x_start = (0.16 + 0.5)
    delta = 0.08 * radius
    w = radius * 0.1

    add_vortex_ring_and_smoke_with_coef(w = w, rad = radius, delta = delta, c = ti.Vector([x_start,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke1, color = ti.Vector([1., 0, 0]), num_samples = 2000, strength= strength)

    add_vortex_ring_and_smoke_with_coef(w = w, rad = radius, delta = delta, c = ti.Vector([(x_start) +x_gap,0.5,0.5]),
            unit_x = ti.Vector([0.,0.,-1.]), unit_y = ti.Vector([0.,1, 0.]),
            pf = X, vf = u, smokef = smoke2, color = ti.Vector([0, 0, 1.]), num_samples = 2000, strength= strength)

    add_fields(smoke1, smoke2, smoke1, 1.0)


def rotate_vector(angle_y, angle_z):
    """Rotate a vector by angle_y around Y-axis and angle_z around Z-axis"""
    # Convert to radians
    vector = np.array([1,0,0])

    #angle_y = np.deg2rad(angle_y)
    #angle_z = np.deg2rad(angle_z)

    # Y-axis rotation matrix
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    # Z-axis rotation matrix
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation
    R = Rz @ Ry
    res = R @ vector
    return res

def orthonormal_basis(n):
    """Given a unit vector n, return two orthonormal vectors u, v such that (u, v, n) is right-handed."""
    # Find a vector not parallel to n
    if abs(n[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])
    
    u = np.cross(n, temp)
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    return ti.Vector(u), ti.Vector(v)

def init_vorts_plume(X, u, smoke1):
    u.fill(0.0)
    # front and back revised
    delta = 0.016
    c = []
    random.seed(42) 
    while(True):
        if(len(c)>=14):
            break
        while(True):
            x = random.random()
            y = random.random()
            z = random.random()
            flag = True
            for i in range(len(c)):
                if(not ((ti.Vector([x,y,z]) - c[i]).norm()>0.25 and 0.05<x<0.4 and 0.15<y<0.85 and 0.15<z<0.85)):
                    flag = False
            if(flag):
                c.append(ti.Vector([x,y,z]))
                break
    c = c[1:]


    """c = [ti.Vector([0.1, 0.2, 0.2]), ti.Vector([0.25, 0.2, 0.8]), 
         ti.Vector([0.2, 0.6, 0.2]),ti.Vector([0.35, 0.7, 0.7]),
         ti.Vector([0.3, 0.7, 0.2]),ti.Vector([0.25, 0.2, 0.7]),
         ti.Vector([0.4, 0.3, 0.6]),ti.Vector([0.35, 0.7, 0.2]),
         ti.Vector([0.1, 0.6, 0.7]),ti.Vector([0.25, 0.2, 0.2]),
         ti.Vector([0.2, 0.2, 0.6]),ti.Vector([0.35, 0.3, 0.6]),
         ti.Vector([0.3, 0.3, 0.6]),ti.Vector([0.25, 0.4, 0.4]),
        ]"""
    
        #ti.Vector([0.55, 0.5, 0.5+0.1]), ti.Vector([0.3, 0.5-0.1, 0.5]), 
        # ti.Vector([0.35, 0.5+0.1, 0.5+0.1]),ti.Vector([0.45, 0.5-0.1, 0.55])
        #]
    unit_n  = [
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
        np.array([1.0,0,0]),np.array([1.0,0,0]),
    ]
    unit_x  =  []
    unit_y  =  []
    for i in range(len(c)):
        u1,u2 = orthonormal_basis(unit_n[i])
        unit_x.append(u1)
        unit_y.append(u2)
    
    strength = [0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003,0.003]
    radius = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
    #strength = [0.010*0.32,0.010*0.32,0.032/3*0.3,0.025/3*0.3,0.028/3*0.3,0.032/3*0.3,0.032/3*0.3]
    #radius = [0.15,0.15,0.1,0.1,0.1,0.1,0.1]
    for i in range(len(c)):
        print(c[i],unit_x[i],unit_y[i],strength[i],radius[i])
        add_vortex_ring_and_smoke(w = strength[i], rad = radius[i], delta = delta, c = c[i],
            unit_x = unit_x[i], unit_y = unit_y[i],
            pf = X, vf = u, smokef = smoke1, color = ti.Vector([1., 0, 0]), num_samples = 500)

    
# initialize four vorts
def init_four_vorts(X, u, smoke1, smoke2):
    u.fill(0.0)

    smoke1.fill(0.)
    smoke2.fill(0.)
    x_offset = 0.16
    y_offset = 0.16
    size = 0.15
    cos45 = ti.cos(math.pi/4)
    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-x_offset,0.5-y_offset, 1]),
        unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke1, color = ti.Vector([1,0.8,0.7]), num_samples = 500)

    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+x_offset,0.5-y_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)
    
    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w = 2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5-x_offset,0.5+y_offset, 1]),
        unit_x = ti.Vector([cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)
    
    add_fields(smoke1, smoke2, smoke1, 1.0)
    smoke2.fill(0.)

    add_vortex_ring_and_smoke(w = -2.e-2, rad = size, delta = 0.024, c = ti.Vector([0.5+x_offset,0.5+y_offset, 1]),
        unit_x = ti.Vector([-cos45,cos45,0.]).normalized(), unit_y = ti.Vector([0.,0.,1.]),
        pf = X, vf = u, smokef = smoke2, color = ti.Vector([1,0.8,0.7]), num_samples = 500)

    add_fields(smoke1, smoke2, smoke1, 1.0)


@ti.kernel
def mask_velocity_by_boundary(boundary_mask:ti.template(),boundary_vel:ti.template(),u_x:ti.template(),u_y:ti.template(),u_z:ti.template()):
    for i,j,k in boundary_mask:
        if(boundary_mask[i,j,k]>=1):
            u_x[i,j,k] = boundary_vel[i,j,k][0]
            u_x[i+1,j,k] = boundary_vel[i,j,k][0]
            u_y[i,j,k] = boundary_vel[i,j,k][1]
            u_y[i,j+1,k] = boundary_vel[i,j,k][1]
            u_z[i,j,k] = boundary_vel[i,j,k][2]
            u_z[i,j,k+1] = boundary_vel[i,j,k][2]


@ti.kernel
def mask_passive_by_boundary(boundary_mask:ti.template(),passive:ti.template()):
    for i,j,k in boundary_mask:
        if(boundary_mask[i,j,k]>=1):
            passive[i,j,k] = 0.0
            w = 0.0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    for kk in range(-1,2):                    
                        iii,jjj,kkk = i+ii,j+jj,k+kk
                        if(valid(iii,jjj,kkk,passive) and boundary_mask[iii,jjj,kkk]<=0):
                            passive[i,j,k]+=passive[iii,jjj,kkk]
                            w+=1
            if(w>0):
                passive[i,j,k]/=w

@ti.kernel
def mask_adj_velocity_by_boundary(boundary_mask:ti.template(),boundary_vel:ti.template(),u_x:ti.template(),u_y:ti.template(),u_z:ti.template()):
    for i,j,k in boundary_mask:
        if(boundary_mask[i,j,k]>=1):
            u_x[i,j,k] = 0.0
            u_x[i+1,j,k] = 0.0
            u_y[i,j,k] = 0.0
            u_y[i,j+1,k] = 0.0
            u_z[i,j,k] = 0.0
            u_z[i,j,k+1] = 0.0

@ti.kernel
def mask_adj_passive_by_boundary(boundary_mask:ti.template(),passive:ti.template()):
    for i,j,k in boundary_mask:
        if(boundary_mask[i,j,k]>=1):
            passive[i,j,k] = 0.0