import taichi as ti
import open3d as o3d
import numpy as np
# ti.init(arch=ti.cuda)
padding = 1
@ti.func
def min3(a, b, c):
    return ti.min(a, ti.min(b, c))

@ti.func
def max3(a, b, c):
    return ti.max(a, ti.max(b, c))

@ti.func
def plane_box_overlap_3d(normal, vert, box_half_size):
    vmin = ti.Vector.zero(float, 3)
    vmax = ti.Vector.zero(float, 3)
    for q in range(3):
        v = vert[q]
        if normal[q] > 0.0:
            vmin[q] = -box_half_size[q] - v
            vmax[q] = box_half_size[q] - v
        else:
            vmin[q] = box_half_size[q] - v
            vmax[q] = -box_half_size[q] - v
    result = False
    if ti.math.dot(normal, vmin) > 0.0:
        result = False
    if ti.math.dot(normal, vmax) >= 0.0:
        result = True
    return result
            
@ti.func
def Axis_Test_X01(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[1] - b * v0[2]
    p2 = a * v2[1] - b * v2[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p2): 
        min = p0
        max = p2
    else:
        min = p2
        max = p0
    rad = fa * box_half_size[1] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result

@ti.func
def Axis_Test_X2(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[1] - b * v0[2];
    p1 = a * v1[1] - b * v1[2];
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[1] + fb * box_half_size[2]
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def Axis_Test_Y02(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = -a * v0[0] + b * v0[2]
    p2 = -a * v2[0] + b * v2[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p2):
        min = p0
        max = p2
    else:
        min = p2
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[2];
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def Axis_Test_Y1(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = -a * v0[0] + b * v0[2]
    p1 = -a * v1[0] + b * v1[2]
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[2];
    if ((min > rad) or (max < -rad)):
        result = False
    return result

@ti.func
def Axis_Test_Z12(a, b, fa, fb, v0, v1, v2, box_half_size):
    p1 = a * v1[0] - b * v1[1]
    p2 = a * v2[0] - b * v2[1]
    min = 0.0
    max = 0.0
    result = True
    if (p2 < p1):
        min = p2
        max = p1
    else:
        min = p1
        max = p2
    
    rad = fa * box_half_size[0] + fb * box_half_size[1];
    if ((min > rad) or (max < -rad)):
        result = False
    return result
@ti.func
def Axis_Test_Z0(a, b, fa, fb, v0, v1, v2, box_half_size):
    p0 = a * v0[0] - b * v0[1];
    p1 = a * v1[0] - b * v1[1];
    min = 0.0
    max = 0.0
    result = True
    if (p0 < p1):
        min = p0
        max = p1
    else:
        min = p1
        max = p0
    rad = fa * box_half_size[0] + fb * box_half_size[1];
    if ((min > rad) or (max < -rad)):
        result = False
    return result
    
@ti.func
def triangle_box_overlap_3d(box_center, box_half_size, x0, x1, x2):
    min = 0.0
    max = 0.0
    v0 = x0 - box_center
    v1 = x1 - box_center
    v2 = x2 - box_center
    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    result = True
    fex = ti.abs(e0[0])
    fey = ti.abs(e0[1])
    fez = ti.abs(e0[2])
    if (not Axis_Test_X01(e0[2], e0[1], fez, fey, v0, v1, v2, box_half_size)):
         result = False
    if (not Axis_Test_Y02(e0[2], e0[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z12(e0[1], e0[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False
    
    
    fex = ti.abs(e1[0]);
    fey = ti.abs(e1[1]);
    fez = ti.abs(e1[2]);
    if (not Axis_Test_X01(e1[2], e1[1], fez, fey, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Y02(e1[2], e1[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z0(e1[1], e1[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False
    
    fex = ti.abs(e2[0]);
    fey = ti.abs(e2[1]);
    fez = ti.abs(e2[2]);
    if (not Axis_Test_X2(e2[2], e2[1], fez, fey, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Y1(e2[2], e2[0], fez, fex, v0, v1, v2, box_half_size)):
        result = False
    if (not Axis_Test_Z12(e2[1], e2[0], fey, fex, v0, v1, v2, box_half_size)):
        result = False

    min = min3(v0[0], v1[0], v2[0])
    max = max3(v0[0], v1[0], v2[0]);
    if ((min > box_half_size[0]) or (max < -box_half_size[0])):
        result = False

    min = min3(v0[1], v1[1], v2[1])
    max = max3(v0[1], v1[1], v2[1])
    if ((min > box_half_size[1]) or (max < -box_half_size[1])):
        result = False

    min = min3(v0[2], v1[2], v2[2])
    max = max3(v0[2], v1[2], v2[2])
    if ((min > box_half_size[2]) or (max < -box_half_size[2])):
        result = False
    normal = ti.math.cross(e0, e1)
    if (not plane_box_overlap_3d(normal, v0, box_half_size)):
        result = False
    return result

@ti.func
def in_dual_cell(pos, center, dx):
    margin = padding * 0.505 * dx;
    return (pos[0] > center[0] - margin) and (pos[0] < center[0] + margin) \
        and (pos[1] > center[1] - margin) and (pos[1] < center[1] + margin) \
        and (pos[2] > center[2] - margin) and (pos[2] < center[2] + margin)


@ti.func
def tri_area(v0, v1, v2):
    return 0.5 * ti.math.cross((v1 - v0), (v2 - v0)).norm()

# def init_mesh(res, scene, dx):
#     xs = np.linspace(dx, res - dx, res)
#     ys = np.linspace(dx, res - dx, res)
#     zs = np.linspace(dx, res - dx, res)
#     xv, yv, zv = np.meshgrid(xs, ys, zs, indexing="ij")
#     xyz = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).transpose().astype(np.float32)
#     occupancy = scene.compute_occupancy(xyz / res).numpy().reshape(res, res, res)
#     return occupancy

# from blender_script import *
# res = 256
# dx = 1.0 / 256
# mesh_dir = "G:/dartmouth/3D_moving_solid/mesh/test_0.obj"
# mesh = o3d.io.read_triangle_mesh(mesh_dir)

# mesh_dir2 = "G:/dartmouth/3D_moving_solid/mesh/test_10.obj"
# mesh2 = o3d.io.read_triangle_mesh(mesh_dir2)

# v = np.asarray(mesh.vertices)
# v2 = np.asarray(mesh2.vertices)

# f = np.asarray(mesh.triangles)

# ti_v = ti.Vector.field(3, float, shape = (v2.shape[0]))
# ti_f = ti.Vector.field(3, int, shape = (f.shape[0]))
# ti_vel = ti.Vector.field(3, float, shape = (v.shape[0]))
# ti_occu = ti.field(float, shape = (res, res, res))
# boundary_mask = ti.field(float, shape = (res, res, res))

# ti_vel.from_numpy(v2 - v)
# print((v2 - v).max())
# ti_v.from_numpy(v)
# ti_f.from_numpy(f)


# mesh2 = o3d.t.geometry.TriangleMesh.from_legacy(mesh2)

# # Create a scene and add the triangle mesh
# scene = o3d.t.geometry.RaycastingScene()
# _ = scene.add_triangles(mesh2)
# occupancy = init_mesh(256, scene, dx)
# ti_occu.from_numpy(occupancy)

@ti.kernel
def calculate_vel(v_old:ti.template(), v_new:ti.template(), vel:ti.template(), dt:float):
    for I in ti.grouped(v_old):
        vel_t = (v_new[I] - v_old[I]) / dt
        # when use bird
        if (vel[I] - vel_t).norm() / dt > 5:
            vel[I].fill(0.0)
        else:
            vel[I] = vel_t
        # vel[I] = vel_t
        
    # for I in ti.grouped(vel):
    #     if (v_new[I] - v_old[I]).norm() < 0.003 or abs((v_new[I] - v_old[I])[0]) < 0.003:
    #         vel[I].fill(0)
    # for I in ti.grouped(vel):
        # if (v_new[I] - v_old[I]).norm()
        #     vel[I].fill(0)
        

            
@ti.kernel
def gen_boundary_mask(field:ti.template(), occu:ti.template(), ti_v:ti.template(), ti_f:ti.template(), ti_vel:ti.template(), dx:float):
#     offset = 0.5 * (1 - ti.Vector.unit(3, dim))
#     occu_offset = ti.Vector.unit(3, dim)
#     half_size = ti.Vector.one(float, 3) * dx * 0.5
#     for I in ti.grouped(field):
#         face_center = (I + offset) * dx
#         area_sum = 0.0
#         if occu[I] == occu[ti.math.clamp(I - occu_offset, 0, res)]:
#             continue
#         velm[I] = 1
#         for J in range(ti_f.shape[0]):
#             velm[I] = 1
#             p0 = ti_v[ti_f[J][0]]
#             p1 = ti_v[ti_f[J][1]]
#             p2 = ti_v[ti_f[J][2]]
            
#             # if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
#             #     continue
#             counts = 0
#             for ii in range(11):
#                 for jj in range(11 - ii):
#                     pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2;
#                     if (in_dual_cell(pt, face_center, dx)):
#                         counts += 1

#             tarea = counts / 66.0 * tri_area(p0, p1, p2)
#             area_sum += tarea
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][0], ti.int32)][dim]
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][1], ti.int32)][dim]
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][2], ti.int32)][dim]
#         if area_sum == 0.0:
#             velm[I] = 0
#             area_sum = 1e-6
#             field[I] = 0
#         field[I] /= area_sum
           


    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    total_cell = 0
    for I in ti.grouped(field):
        inter = 0.0
        # field[I] = 
        face_center = (I + offset) * dx
        area_sum = 0.0
        
        for J in range(ti_f.shape[0]):
            p0 = ti_v[ti_f[J][0]]
            p1 = ti_v[ti_f[J][1]]
            p2 = ti_v[ti_f[J][2]]
            
            if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
                inter += 1
                continue
            # occu[I] = 1
            counts = 0
            for ii in range(11):
                for jj in range(11 - ii):
                    pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2;
                    if (in_dual_cell(pt, face_center, dx)):
                        counts += 1

            tarea = counts / 66.0 * tri_area(p0, p1, p2)
            area_sum += tarea
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][0], ti.int32)]
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][1], ti.int32)]
            field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][2], ti.int32)]
        if area_sum == 0.0:
            # field[I].fill(0.0)
            area_sum = 1e-6
        field[I] /= area_sum
            
#     for I in ti.grouped(field):
#         if occu[I] >= 1 and intersect[I] != 1:
#             nearest = 1e20
#             # loc = (I * dx)
#             for a in ti.ndrange(-5, 5):
#                 for b in ti.ndrange(-5, 5):
#                     for c in ti.ndrange(-5, 5):
#                         offset = ti.Vector([a, b, c])
#                         if intersect[I + ti.cast(offset, ti.int32)] >= 1 and offset.norm() * dx < nearest:
#                             nearest = offset.norm() * dx
#                             field[I] = intersect[I + ti.cast(offset, ti.int32)]
            
    

@ti.func
def point_segment_distance(x0 : ti.template(), x1 : ti.template(), x2 : ti.template()):
    dx = x2 - x1
    m2 = ti.math.dot(dx, dx)
    s12 = ti.math.dot(dx, x2 - x0) / m2
    if s12 < 0.0:
        s12 = 0.0
    elif s12 > 1.0:
        s12 = 1.0
    return (x0 - (s12 * x1 + (1 - s12) * x2)).norm() 

@ti.func
def point_triangle_distance(x0 : ti.template(), x1 : ti.template(), x2 : ti.template(), x3 : ti.template()):
    # first find barycentric coordinates of closest point on infinite plane
    result = 0.0
    x13 = x1 - x3
    x23 = x2 - x3
    x03 = x0 - x3
    m13 = ti.math.dot(x13, x13)
    m23 = ti.math.dot(x23, x23)
    d = ti.math.dot(x13, x23)
    invdet = 1.0 / ti.max(m13 * m23 - d * d, 1e-30);
    a = ti.math.dot(x13, x03)
    b = ti.math.dot(x23, x03)
    # the barycentric coordinates themselves
    w23 = invdet * (m23 * a - d * b);
    w31 = invdet * (m13 * b - d * a);
    w12 = 1 - w23 - w31;
    if (w23 >= 0 and w31 >= 0 and w12 >= 0):
        # if we're inside the triangle
        result = (x0 - (w23 * x1 + w31 * x2 + w12 * x3)).norm()
    else:
        # we have to clamp to one of the edges
        if w23 > 0:
            #this rules out edge 2-3 for us
            result = ti.min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x1, x3))
        elif w31 > 0:
            # this rules out edge 1-3
            result = ti.min(point_segment_distance(x0, x1, x2), point_segment_distance(x0, x2, x3))
        else:
            # w12 must be >0, ruling out edge 1-2
            result = ti.min(point_segment_distance(x0, x1, x3), point_segment_distance(x0, x2, x3));
    return result

        
@ti.kernel
def update_boundary_mask(occu:ti.template(), ti_v:ti.template(), ti_f:ti.template(), dx:float):
#     offset = 0.5 * (1 - ti.Vector.unit(3, dim))
#     occu_offset = ti.Vector.unit(3, dim)
#     half_size = ti.Vector.one(float, 3) * dx * 0.5
#     for I in ti.grouped(field):
#         face_center = (I + offset) * dx
#         area_sum = 0.0
        
#         for J in range(ti_f.shape[0]):
#             p0 = ti_v[ti_f[J][0]]
#             p1 = ti_v[ti_f[J][1]]
#             p2 = ti_v[ti_f[J][2]]
            
#             if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
#                 continue
#             counts = 0
#             for ii in range(11):
#                 for jj in range(11 - ii):
#                     pt = ii / 10.0 * p0 + jj / 10.0 * p1 + (10 - ii - jj) / 10.0 * p2;
#                     if (in_dual_cell(pt, face_center, dx)):
#                         counts += 1

#             tarea = counts / 66.0 * tri_area(p0, p1, p2)
#             area_sum += tarea
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][0], ti.int32)][dim]
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][1], ti.int32)][dim]
#             field[I] += 1.0 / 3 * tarea * ti_vel[ti.cast(ti_f[J][2], ti.int32)][dim]
#         if area_sum == 0.0:
#             area_sum = 1e-6
#         field[I] /= area_sum


    offset = 0.5 * ti.Vector.one(float, 3)
    # occu_offset = ti.Vector.unit(3, dim)
    half_size = ti.Vector.one(float, 3) * dx * 0.5
    for I in ti.grouped(occu):
        face_center = (I + offset) * dx
        area_sum = 0.0
        
        for J in range(ti_f.shape[0]):
            p0 = ti_v[ti_f[J][0]]
            p1 = ti_v[ti_f[J][1]]
            p2 = ti_v[ti_f[J][2]]
            
            if not triangle_box_overlap_3d(face_center, half_size, p0, p1, p2):
                continue
            occu[I] = ti.cast(1, ti.int32)
        
        

# check_mesh_box_intersect(boundary_mask, 0, ti_v, ti_f, ti_vel, 1/256)
# check_mesh_box_intersect(boundary_mask, ti_occu, 0, ti_v, ti_f, ti_vel, 1/256)
# np.save("G:/dartmouth/3D_moving_solid/test", boundary_mask.to_numpy())
# np.save("G:/dartmouth/3D_moving_solid/test_sign", (occupancy < 0))