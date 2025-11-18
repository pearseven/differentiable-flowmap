dim = 2
res_x = 256
res_y = 256
dx = 1.0 /res_y
dpi_vor = 32
reinit_every = 10
CFL = 1.0

exp_name = f"optimize_eigen_fluid_dragon_passive"
BFECC_clamp = False
RK_number = 2
use_short_BFECC = False

act_dt = 0.01
frame_per_step = 10
total_steps = 100
output_image_frame = False
add_passive_scalar = True
poisson_output_log = False
backward_u = True
add_control_force = True
control_num = 3000
viscosity = 0.0

print_time = False