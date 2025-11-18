dim = 2
res_x = 256
res_y = 256
dx = 1.0 /res_y
dpi_vor = 512
reinit_every = 10
CFL = 1.0

exp_name = f"optimize_passive_16_for_8_with_obstacles"
BFECC_clamp = False
RK_number = 2
use_short_BFECC = False

act_dt = 0.005
frame_per_step = 10
total_steps = 400
sub_steps = 100
sub_iters = 120

sub_optimize=False
output_image_frame = False
add_passive_scalar = True
poisson_output_log = True
backward_u = True
add_control_force = False
viscosity = 0.0

print_time = False
