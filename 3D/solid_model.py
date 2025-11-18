from hyperparameters import *
import matplotlib.pyplot as plt
from levelset import *
from mesh_interpolation import *
from taichi_utils import *
import math

def draw_3D_point(x,y,z):
    ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.scatter(x, y, z, c = 'r')   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色

    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    ax.set_zlabel('Z')  # 设置z坐标轴
    plt.savefig("./tem.png")

def stretch_model(v,name):
    max_x,min_x=np.max(v[:, 0]),np.min(v[:, 0])
    max_y,min_y=np.max(v[:, 1]),np.min(v[:, 1])
    max_z,min_z=np.max(v[:, 2]),np.min(v[:, 2])
    print(max_x,min_x,max_y,min_y,max_z,min_z)
    print("shape of vertex array",v.shape)
    # first normalize the nodel
    ratio_y=(max_y-min_y)/(max_z-min_z)
    ratio_x=(max_x-min_x)/(max_z-min_z)
    v[:, 0]=((v[:, 0]-min_x)/(max_x-min_x)-0.5)*ratio_x
    v[:, 1]=((v[:, 1]-min_y)/(max_y-min_y)-0.5)*ratio_y
    v[:, 2]=((v[:, 2]-min_z)/(max_z-min_z)-0.5)

    print("name",name)
    if(name == "bunny"):
        v[:, 0]*= 0.5
        v[:, 1]*= 0.5
        v[:, 2]*= 0.5
        v[:, 0]+= 0.5
        v[:, 1]+= 0.5
        v[:, 2]+= 0.5
    elif(name == "armadillo"):
        v[:, 0]*= 0.5/0.7115179
        v[:, 1]*= 0.5/0.7115179
        v[:, 2]*= 0.5/0.7115179
        v[:, 0]+= 0.5
        v[:, 1]+= 0.5
        v[:, 2]+= 0.5
    
    max_x,min_x=np.max(v[:, 0]),np.min(v[:, 0])
    max_y,min_y=np.max(v[:, 1]),np.min(v[:, 1])
    max_z,min_z=np.max(v[:, 2]),np.min(v[:, 2])
    print(max_x,min_x,max_y,min_y,max_z,min_z)

"""def load_model(model_name,stretch_name, boundary_mask,dx,verbose = True):
    mesh = o3d.io.read_triangle_mesh(model_name)#,boundary_mask)
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    max_x,min_x=np.max(v[:, 0]),np.min(v[:, 0])
    max_y,min_y=np.max(v[:, 1]),np.min(v[:, 1])
    max_z,min_z=np.max(v[:, 2]),np.min(v[:, 2])
    if(verbose):
        print(max_x,min_x,max_y,min_y,max_z,min_z)
        print("shape of vertex array",v.shape,"shape of face array",f.shape)
    # first normalize the nodel
    ratio_y=(max_y-min_y)/(max_z-min_z)
    ratio_x=(max_x-min_x)/(max_z-min_z)
    v[:, 0]=((v[:, 0]-min_x)/(max_x-min_x)-0.5)*ratio_x
    v[:, 1]=((v[:, 1]-min_y)/(max_y-min_y)-0.5)*ratio_y
    v[:, 2]=((v[:, 2]-min_z)/(max_z-min_z)-0.5)
    stretch_model(v,stretch_name)
    mesh_v = ti.Vector.field(3, float, shape = (v.shape[0]))
    mesh_f = ti.Vector.field(3, int, shape = (f.shape[0]))
    mesh_v.from_numpy(v)
    mesh_f.from_numpy(f)
    draw_3D_point(v[:, 0],v[:, 1],v[:, 2])
    level_set = LevelSet([res_x, res_y, res_z], dx, mesh_f)
    level_set.redistance(mesh_v)
    level_set.fill_occu1(boundary_mask)"""

def filter_model(bm, center,name):
    if(name == "bunny"):
        mask = center[..., 1] < 0.18
        bm[mask] = 0
        return bm  # 可选：返回修改后的 bm

def load_model2(model_name,stretch_name, boundary_mask,dx,verbose = True):
    mesh = o3d.io.read_triangle_mesh(model_name)#,boundary_mask)
    vertices = np.asarray(mesh.vertices)
    stretch_model(vertices,stretch_name)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    mesh.compute_vertex_normals()

    # 转换为 Open3D Tensor mesh（RaycastingScene 需要）
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # 创建 Raycasting 场景
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    # 测试点（N x 3 numpy array）
    xs = (np.arange(res_x) + 0.5) / res_x
    ys = (np.arange(res_y) + 0.5) / res_y
    zs = (np.arange(res_z) + 0.5) / res_z

    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
    cell_centers = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)

    # 转为 Tensor 并查询
    queries = o3d.core.Tensor(cell_centers, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(queries)

    # 判断内部：SDF < 0
    inside_mask = signed_distance.numpy() < 0
    print(np.sum(inside_mask),"inside_mask")

    inside_mask = inside_mask.reshape(res_x,res_y,res_z)
    cell_centers = cell_centers.reshape(res_x,res_y,res_z, 3)
    boundary_mask.fill(0)
    bm = boundary_mask.to_numpy()
    bm[inside_mask] = 1
    filter_model(bm, cell_centers ,stretch_name)
    boundary_mask.from_numpy(bm)

    
    
    # 输出结果
    #for pt, inside in zip(cell_centers, inside_mask):
    #    print(f"Point {pt} is inside mesh? {inside}")
