from taichi_utils import *
from hyperparameters import *
import taichi as ti
###########################################################################################
##################      1. Test the correctness of Euler        ###########################
###########################################################################################
@ti.func
def angular_vel_func(r, rad, strength):
    r = r + 1e-6
    linear_vel = strength * 1./r * (1.-ti.exp(-(r**2)/(rad**2)))
    return 1./r * linear_vel


@ti.kernel
def vortex_vel_func_with_coef(vf: ti.template(), pf: ti.template(), w:float):
    c = ti.Vector([0.5, 0.5])
    for i, j in vf:
        p = pf[i, j] - c
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j].y = p.x
        vf[i, j].x = -p.y
        vf[i, j] *= angular_vel_func(r, 0.02, -0.01)*w

@ti.kernel
def four_vortex_vel_func_with_coef2(vf: ti.template(), pf: ti.template(), w1:float, w2:float, w3:float, w4:float):
    c1 = ti.Vector([0.35, 0.62])
    c2 = ti.Vector([0.65, 0.38])
    c3 = ti.Vector([0.65, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    cs = [c1, c2, c3, c4]
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w1 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w2 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w3 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w4 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition

@ti.kernel
def two_vortex_vel_func_with_coef(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.35, 0.62])
    c2 = ti.Vector([0.65, 0.38])
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) *ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition

@ti.kernel
def eight_vortex_vel_func_with_coef2(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.35, 0.62])
    c2 = ti.Vector([0.65, 0.38])
    c3 = ti.Vector([0.65, 0.74])
    c4 = ti.Vector([0.25, 0.26])

    c5 = ti.Vector([0.45, 0.22])
    c6 = ti.Vector([0.45, 0.18])
    c7 = ti.Vector([0.35, 0.74])
    c8 = ti.Vector([0.85, 0.26])

    cs = [c1, c2, c3, c4]
    w1 = -0.5
    w2 = 0.5
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w1 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.06, -0.01) * w2 * ti.Vector([-p.y, p.x])*1.5
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w1 * ti.Vector([-p.y, p.x])*1.0
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w2 * ti.Vector([-p.y, p.x])*3.5
        vf[i, j] += addition



        p = pf[i, j] - c5
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.01, -0.01) * w1 * ti.Vector([-p.y, p.x])*0.5
        #vf[i, j] += addition
        # c2
        p = pf[i, j] - c6
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.06, -0.01) * w2 * ti.Vector([-p.y, p.x])*2.7
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c7
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w1 * ti.Vector([-p.y, p.x])*1.8
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c8
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w2 * ti.Vector([-p.y, p.x])*3.5
        vf[i, j] += addition



@ti.kernel
def four_vortex_vel_func_with_pos_coef(
    vf: ti.template(), 
    pf: ti.template(), 
    w1:float, w2:float, w3:float, w4:float,
    c1_x:float, c2_x:float, c3_x:float, c4_x:float,
    c1_y:float, c2_y:float, c3_y:float, c4_y:float   
):
    c1 = ti.Vector([c1_x,c1_y])
    c2 = ti.Vector([c2_x,c2_y])
    c3 = ti.Vector([c3_x,c3_y])
    c4 = ti.Vector([c4_x,c4_y])
    cs = [c1, c2, c3, c4]
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w1 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w2 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w3 * ti.Vector([-p.y, p.x])*2.5
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.05, -0.01) * w4 * ti.Vector([-p.y, p.x])*2.5

@ti.kernel
def sixteen_vortex_vel_func_with_pos_coef(
    vf: ti.template(), 
    pf: ti.template(), 
    theta:   ti.types.vector(64, float)
):
    vf.fill(0.0)
    for i, j in vf:
        for k in range(16):
            c = ti.Vector([theta[16+k], theta[32+k]])
            w = theta[k]
            rr = theta[48+k]
            p = pf[i, j] - c
            r = ti.sqrt(p.x * p.x + p.y * p.y)
            addition = angular_vel_func(r, 0.05*rr, -0.01) * w * ti.Vector([-p.y, p.x])*2.5
            vf[i, j] += addition

@ti.kernel
def four_vortex_vel_func_with_coef(
    vf: ti.template(), 
    pf: ti.template(), w1:float, w2:float, w3:float, w4:float):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    cs = [c1, c2, c3, c4]
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition

@ti.kernel
def simple_passive(passive: ti.template(), pf: ti.template()):
    for i, j in passive:
        passive[i, j] = (pf[i, j].x+pf[i, j].y)/2
        
        #if(pf[i, j].y>0.2 and pf[i, j].y<0.8 and pf[i, j].x>0.3 and pf[i, j].x<0.4):
        #    passive[i, j] =  1.0
        #else:
        #    passive[i, j] =  0.0

@ti.kernel
def vortex_vel_func(vf: ti.template(), pf: ti.template()):
    c = ti.Vector([0.5, 0.5])
    for i, j in vf:
        p = pf[i, j] - c
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j].y = p.x
        vf[i, j].x = -p.y
        vf[i, j] *= angular_vel_func(r, 0.02, -0.01)

@ti.kernel
def set_boundary_mask(boundary_mask:ti.template(),boundary_vel:ti.template()):
    boundary_mask.fill(0.0)
    boundary_vel.fill(0.0)


@ti.kernel
def set_circle_boundary_mask(boundary_mask:ti.template(),boundary_vel:ti.template()):
    boundary_vel.fill(0.0)
    for i,j in boundary_mask:
        p = ti.Vector([i+0.5,j+0.5])*dx
        if abs(p[0]-0.5)<0.05 and abs(p[1]-0.5)<0.05:
            boundary_mask[i,j] = 1.0
        else:
            boundary_mask[i,j] = 0.0
        

@ti.kernel
def mask_velocity_by_boundary(boundary_mask:ti.template(),boundary_vel:ti.template(),u_x:ti.template(),u_y:ti.template()):
    for i,j in boundary_mask:
        if(boundary_mask[i,j]>=1):
            u_x[i,j] = boundary_vel[i,j][0]
            u_x[i+1,j] = boundary_vel[i,j][0]
            u_y[i,j] = boundary_vel[i,j][1]
            u_y[i,j+1] = boundary_vel[i,j][1]

@ti.kernel
def mask_passive_by_boundary(boundary_mask:ti.template(),passive:ti.template()):
    for i,j in boundary_mask:
        if(boundary_mask[i,j]>=1):
            passive[i,j] = 0.0
            w = 0.0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    iii,jjj = i+ii,j+jj
                    if(valid(iii,jjj,passive) and boundary_mask[iii,jjj]<=0):
                        passive[i,j]+=passive[iii,jjj]
                        w+=1
            if(w>0):
                passive[i,j]/=w

@ti.kernel
def mask_adj_velocity_by_boundary(boundary_mask:ti.template(),boundary_vel:ti.template(),u_x:ti.template(),u_y:ti.template()):
    for i,j in boundary_mask:
        if(boundary_mask[i,j]>=1):
            u_x[i,j] = 0.0
            u_x[i+1,j] = 0.0
            u_y[i,j] = 0.0
            u_y[i,j+1] = 0.0

@ti.kernel
def mask_adj_passive_by_boundary(boundary_mask:ti.template(),passive:ti.template()):
    for i,j in boundary_mask:
        if(boundary_mask[i,j]>=1):
            passive[i,j] = 0.0

# @ti.kernel
# def mask_by_boundary(boundary_mask:ti.template(),boundary_vel:ti.template(),u_x:ti.template(),u_y:ti.template()):
#     for i,j in boundary_mask:
#         if(boundary_mask[i,j]>=1):
#             u_x[i,j] = boundary_vel[i,j][0]
#             u_x[i+1,j] = boundary_vel[i,j][0]
#             u_y[i,j] = boundary_vel[i,j][1]
#             u_y[i,j+1] = boundary_vel[i,j][1]
##########################################################################################
##########################################################################################

@ti.kernel
def leapfrog_vel_func(vf: ti.template(), pf: ti.template()):
    c1 = ti.Vector([0.25, 0.62])
    c2 = ti.Vector([0.25, 0.38])
    c3 = ti.Vector([0.25, 0.74])
    c4 = ti.Vector([0.25, 0.26])
    cs = [c1, c2, c3, c4]
    w1 = -0.5
    w2 = 0.5
    w3 = -0.5
    w4 = 0.5
    for i, j in vf:
        # c1
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w1 * ti.Vector([-p.y, p.x])
        vf[i, j] = addition
        # c2
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w2 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c3
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w3 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition
        # c4
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.02, -0.01) * w4 * ti.Vector([-p.y, p.x])
        vf[i, j] += addition


