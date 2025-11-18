from taichi_utils import *
from hyperparameters import *
import taichi as ti
import numpy as np
from PIL import Image
###########################################################################################
##################      1. Test the correctness of Euler        ###########################
###########################################################################################

@ti.func
def taylor_x(x, t, viscosity):
    return ti.sin(2 * ti.math.pi* x[0]) * ti.cos(2 * ti.math.pi* x[1]) * ti.exp(-8 * ti.math.pi**2 * viscosity * t)

@ti.func
def taylor_y(x, t, viscosity):
    return -ti.cos(2 * ti.math.pi* x[0]) * ti.sin(2 * ti.math.pi* x[1]) * ti.exp(-8 * ti.math.pi**2 * viscosity * t)

@ti.kernel
def set_cylinder_obstacles(boundary_mask: ti.template(), boundary_vel: ti.template()):
    # 清空掩码
    boundary_mask.fill(0.0)
    boundary_vel.fill(0.0)
    
    # 定义圆柱参数
    cyl1_pos_x = 0.3
    cyl1_pos_y = 0.75
    cyl1_radius = 0.05
    
    cyl2_pos_x = 0.4
    cyl2_pos_y = 0.5
    cyl2_radius = 0.05
    
    cyl3_pos_x = 0.3
    cyl3_pos_y = 0.25
    cyl3_radius = 0.05
    
    # 获取域的维度
    nx, ny = boundary_mask.shape
    
    # 计算圆柱的网格单位属性
    cyl1_x = int(cyl1_pos_x * nx)
    cyl1_y = int(cyl1_pos_y * ny)
    cyl1_r = int(cyl1_radius * nx)
    
    cyl2_x = int(cyl2_pos_x * nx)
    cyl2_y = int(cyl2_pos_y * ny)
    cyl2_r = int(cyl2_radius * nx)
    
    cyl3_x = int(cyl3_pos_x * nx)
    cyl3_y = int(cyl3_pos_y * ny)
    cyl3_r = int(cyl3_radius * nx)
    
    # 创建圆柱形障碍物
    for i, j in boundary_mask:
        # 检查点是否在任一圆柱内
        dist1 = ti.sqrt((i - cyl1_x)**2 + (j - cyl1_y)**2)
        dist2 = ti.sqrt((i - cyl2_x)**2 + (j - cyl2_y)**2)
        dist3 = ti.sqrt((i - cyl3_x)**2 + (j - cyl3_y)**2)
        
        if dist1 <= cyl1_r or dist2 <= cyl2_r or dist3 <= cyl3_r:
            boundary_mask[i, j] = 1.0


@ti.kernel
def set_airplane_boundary(boundary_mask: ti.template(), boundary_vel: ti.template()):
    boundary_vel.fill(0.0)
    
    # Airplane position and size
    x_center = 0.5
    y_center = 0.5
    scale = 0.25
    
    for i, j in boundary_mask:
        p = ti.Vector([i+0.5, j+0.5]) * dx
        
        # Transform to local coordinates
        x_rel = (p[0] - x_center) / scale
        y_rel = (p[1] - y_center) / scale
        
        is_boundary = False
        
        # Fuselage (elongated body with sharper nose)
        if -0.8 <= x_rel <= 0.8 and -0.15 <= y_rel <= 0.15:
            # Create a much sharper nose
            if x_rel > 0.5:
                # Sharper nose taper
                max_width = 0.15 * (1.0 - (x_rel - 0.5) * (x_rel - 0.5) * 4.0)
                if abs(y_rel) < max_width:
                    is_boundary = True
            # Tail section
            elif x_rel < -0.5:
                max_width = 0.15 * (1.0 - (x_rel + 0.5) * (x_rel + 0.5))
                if abs(y_rel) < max_width:
                    is_boundary = True
            # Middle fuselage
            elif -0.5 <= x_rel <= 0.5:
                if abs(y_rel) < 0.15:
                    is_boundary = True
        
        # Main wings (symmetric and pointed)
        if -0.1 <= x_rel <= 0.4:
            # For upper and lower wings
            if 0.1 <= abs(y_rel) <= 0.8:
                # Calculate wing edge boundary - more pointed at tip
                tip_sharpness = 0.7  # Higher value = more pointed
                wing_edge = 0.4 - (abs(y_rel) - 0.1) * tip_sharpness
                if -0.1 <= x_rel <= wing_edge:
                    is_boundary = True
        
        # Tail wings (horizontal stabilizers) - symmetric and pointed
        if -0.8 <= x_rel <= -0.5:
            # Both upper and lower stabilizers
            if 0.1 <= abs(y_rel) <= 0.4:
                stab_edge = -0.5 - (abs(y_rel) - 0.1) * 0.75
                if x_rel >= stab_edge:
                    is_boundary = True
        
        # Vertical stabilizer (tail fin)
        if -0.8 <= x_rel <= -0.5 and abs(y_rel) <= 0.4:
            # Make it centered and symmetrical
            if x_rel >= -0.8 + (0.4 - abs(y_rel)) * 0.8:
                is_boundary = True
        
        # Set the boundary mask based on our calculations
        if is_boundary:
            boundary_mask[i, j] = 1.0
        else:
            boundary_mask[i, j] = 0.0

@ti.kernel
def set_up_taylor(u_x:ti.template(), u_y:ti.template(), viscosity:float, dt:float):
    for I in ti.grouped(u_x):
        p = (I+0.5)*dx
        u_x[I] = taylor_x(p, dt, viscosity)
    for I in ti.grouped(u_y):
        p = (I+0.5)*dx
        u_y[I] = taylor_y(p, dt, viscosity)



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
def eight_vortex_vel_func(vf: ti.template(), pf: ti.template()):
    # Fixed vortex positions - all positioned strategically around the airplane
    # These positions assume the airplane is centered around (0.5, 0.5)
    
    # Vortices near the wings
    c1 = ti.Vector([0.75, 0.45])  # Above right wing
    c2 = ti.Vector([0.75, 0.15])  # Below right wing
    c3 = ti.Vector([0.25, 0.85])  # Above left wing
    c4 = ti.Vector([0.25, 0.55])  # Below left wing
    
    # Vortices near the tail
    c5 = ti.Vector([0.15, 0.75])  # Above tail
    c6 = ti.Vector([0.15, 0.65])  # Below tail
    
    # Vortices in front of the airplane
    c7 = ti.Vector([0.85, 0.4])   # Above nose
    c8 = ti.Vector([0.85, 0.2])   # Below nose
    
    # Vortex strengths
    w_clockwise = 0.5
    w_counterclockwise = -0.5
    
    # Vortex parameters
    core_size = 0.03
    cutoff = -0.01
    
    for i, j in vf:
        # Initialize with background flow
        vf[i, j] = ti.Vector([-0.2, 0.0])
        
        # Wing vortices
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_clockwise * ti.Vector([-p.y, p.x]) * 2.0
        
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_counterclockwise * ti.Vector([-p.y, p.x]) * 2.5
        
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_clockwise * ti.Vector([-p.y, p.x]) * 3.0
        
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_counterclockwise * ti.Vector([-p.y, p.x]) * 2.0
        
        # Tail vortices
        p = pf[i, j] - c5
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_clockwise * ti.Vector([-p.y, p.x]) * 1.8
        
        p = pf[i, j] - c6
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_counterclockwise * ti.Vector([-p.y, p.x]) * 1.8
        
        # Front vortices
        p = pf[i, j] - c7
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_clockwise * ti.Vector([-p.y, p.x]) * 2.2
        
        p = pf[i, j] - c8
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        vf[i, j] += angular_vel_func(r, core_size, cutoff) * w_counterclockwise * ti.Vector([-p.y, p.x]) * 2.2
       

@ti.kernel
def eight_vortex_vel_func_with_coef2(vf: ti.template(), pf: ti.template()):
    # Top spiral cluster
    c1 = ti.Vector([0.90, 0.8])      # Central vortex in upper cluster
    c2 = ti.Vector([0.72, 0.15])     # Left vortex in upper cluster
    c3 = ti.Vector([0.9, 0.6])     # Right vortex in upper cluster
    c4 = ti.Vector([0.62, 0.53])     # Bottom vortex in upper cluster
    
    # Bottom spiral cluster    
    c5 = ti.Vector([0.62, 0.43])       # Central vortex in lower cluster
    c6 = ti.Vector([0.9, 0.36])     # Left vortex in lower cluster
    c7 = ti.Vector([0.72, 0.25])     # Right vortex in lower cluster
    c8 = ti.Vector([0.9, 0.7])      # Bottom vortex in lower cluster
    
    # All blue vortices since image only shows blue
    w_blue = 0.5
    w_red = -0.5  # Changed to match blue
    
    for i, j in vf:
        vf[i, j] = ti.Vector([-0.2, -0.0])  # Initialize velocity field
        
        # Upper cluster vortices
        p = pf[i, j] - c1
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w_blue * ti.Vector([-p.y, p.x]) * 2.0
        vf[i, j] += addition
        
        p = pf[i, j] - c2
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w_red * ti.Vector([-p.y, p.x]) * 2.5
        vf[i, j] += addition
        
        p = pf[i, j] - c3
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w_blue * ti.Vector([-p.y, p.x]) * 4.
        vf[i, j] += addition
        
        p = pf[i, j] - c4
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w_blue * ti.Vector([-p.y, p.x]) * 2.0
        vf[i, j] += addition
        
        # Lower cluster vortices
        p = pf[i, j] - c5
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w_red * ti.Vector([-p.y, p.x]) * 2.0
        vf[i, j] += addition
        
        p = pf[i, j] - c6
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w_red * ti.Vector([-p.y, p.x]) * 4.
        vf[i, j] += addition
        
        p = pf[i, j] - c7
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.03, -0.01) * w_blue * ti.Vector([-p.y, p.x]) * 2.2
        vf[i, j] += addition
        
        p = pf[i, j] - c8
        r = ti.sqrt(p.x * p.x + p.y * p.y)
        addition = angular_vel_func(r, 0.04, -0.01) * w_red * ti.Vector([-p.y, p.x]) * 2.4
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

def load_image_to_density(path):
    # Load image and convert to RGB
    img = Image.open(path).convert("RGB")
    
    # Get passive field dimensions (will be passed in the simple_passive function)
    # Resize the image to match passive's dimensions
    # (This will be adjusted in the simple_passive function)
    
    # Get image as numpy array
    img_array = np.array(img).astype(np.float32)
    # Create empty density array
    H, W, _ = img_array.shape
    density = np.zeros((H, W), dtype=np.float32)
    
    # Create a mask for red and blue regions
    red_mask = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    
    # Process each pixel to identify red and blue regions
    for i in range(H):
        for j in range(W):
            # Get RGB values (normalized to [0,1])
            r = img_array[i, j, 0] / 255.0
            g = img_array[i, j, 1] / 255.0
            b = img_array[i, j, 2] / 255.0
            
            # Detect color dominance
            red_score = r - (b + g) / 3
            blue_score = b - (r + g) / 3
            
            # Create masks
            if blue_score > 0.1 and blue_score > red_score:
                blue_mask[i, j] = True
            elif red_score > 0.1 and red_score > blue_score:
                red_mask[i, j] = True
    
    # Compute distance from red and blue regions
    from scipy.ndimage import distance_transform_edt
    
    # Distance from red regions (for areas outside red)
    red_distance = distance_transform_edt(~red_mask)
    # Distance from blue regions (for areas outside blue)
    blue_distance = distance_transform_edt(~blue_mask)
    
    # Calculate narrowband values
    max_red_dist = np.max(red_distance)
    max_blue_dist = np.max(blue_distance)
    
    # Normalize distances
    norm_red_dist = red_distance / max_red_dist if max_red_dist > 0 else red_distance
    norm_blue_dist = blue_distance / max_blue_dist if max_blue_dist > 0 else blue_distance
    
    # Compute density based on distance fields
    # Red areas get value 1.0
    density[red_mask] = 1.0
    # Blue areas get value 0.0
    density[blue_mask] = 0.0
    
    # For transitional areas, use narrowband approach
    transition_mask = ~(red_mask | blue_mask)
    # Calculate ratio of distances to determine value (closer to red = closer to 1.0)
    transition_ratio = norm_blue_dist / (norm_blue_dist + norm_red_dist + 1e-10)
    # Apply power function for sharper or smoother transition
    density[transition_mask] = transition_ratio[transition_mask] ** 0.5
    
    return density

@ti.kernel
def simple_passive_kernel(passive1: ti.template(), passive2: ti.template(), 
                          passive3: ti.template(), pf: ti.template(), 
                          density: ti.template()):
    for i, j in passive1:
        # Fill each passive field with corresponding density values from different channels
        passive1[i, j] = density[i, j, 0]  # R channel
        passive2[i, j] = density[i, j, 1]  # G channel
        passive3[i, j] = density[i, j, 2]  # B channel

def simple_passive(passive1, passive2, passive3, pf, image_path="data/gradient.jpg"):
    # Get passive dimensions (assuming all passive fields have the same shape)
    passive_shape = passive1.shape
    H, W = passive_shape
   
    # Load and resize the image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((W, H), Image.BICUBIC)  # Use BICUBIC for better quality scaling
   
    # Convert to numpy and normalize to [0,1]
    img_array = np.array(img).astype(np.float32) / 255.0
   
    # Create 3-channel density array directly from normalized image
    density_np = img_array
   
    # Initialize 3-channel density field
    density = ti.field(dtype=ti.f32, shape=(H, W, 3))
   
    # Copy numpy array to Taichi field
    density.from_numpy(density_np)
   
    # Call kernel to write to passive fields
    simple_passive_kernel(passive1, passive2, passive3, pf, density)
   
    # Return all three passive fields as numpy arrays
    return passive1.to_numpy(), passive2.to_numpy(), passive3.to_numpy()


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


