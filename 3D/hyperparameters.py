case = 11


if case == 2:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"test_single_vortex"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.05
    frame_per_step = 10
    total_steps = 1000
    output_image_frame = True
    add_passive_scalar = False
    poisson_output_log = False
    backward_u = False
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = True

elif case == 3:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_single_vortex"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.1
    frame_per_step = 10
    total_steps = 400
    output_image_frame = True
    add_passive_scalar = False
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False

elif case == 4:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"test_leapfrog2"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.05
    frame_per_step = 10
    total_steps = 400
    output_image_frame = True
    add_passive_scalar = False
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False


elif case == 5:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_single_vortex2"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.2
    frame_per_step = 10
    total_steps = 400
    output_image_frame = True
    add_passive_scalar = False
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False

elif case == 6:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_single_vortex3"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.2
    frame_per_step = 10
    total_steps = 30
    output_image_frame = True
    add_passive_scalar = False
    poisson_output_log =  True
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False


elif case == 7:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_leapfrog_passive_stage1"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.1
    frame_per_step = 10
    total_steps = 10
    output_image_frame = True
    add_passive_scalar = True
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False


elif case == 8:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_leapfrog_passive_stage2"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.1
    frame_per_step = 10
    total_steps = 100
    output_image_frame = True
    add_passive_scalar = True
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False

elif case == 9:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_plume_passive_stage1"
    BFECC_clamp = False
    RK_number = 2
    use_short_BFECC = False
    
    act_dt = 0.1
    frame_per_step = 10
    total_steps = 10
    output_image_frame = True
    add_passive_scalar = True
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False

elif case == 10:
    dim = 2
    res_x = 256
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 10
    CFL = 1.0

    exp_name = f"optimize_plume_passive_stage2"
    BFECC_clamp = True
    RK_number = 4
    use_short_BFECC = True
    
    act_dt = 0.1
    frame_per_step = 10
    total_steps = 1000
    output_image_frame = True
    add_passive_scalar = True
    poisson_output_log = False
    backward_u = True
    add_control_force = False
    viscosity = 0.0
    control_num = 0

    print_time = False

elif case == 11:
    dim = 3
    res_x = 128
    res_y = 128
    res_z = 128

    dx = 1.0 /res_y
    dpi_vor = 512
    reinit_every = 12
    CFL = 1.0

    exp_name = f"optimize_shape_G_R"
    BFECC_clamp = False
    RK_number = 4
    use_short_BFECC = False
    
    act_dt = 0.0085#0.02
    frame_per_step = 12
    total_steps = 120
    output_image_frame = True
    add_passive_scalar = True
    poisson_output_log = False
    backward_u = True
    add_control_force = True
    control_num = 3000
    viscosity = 0.0

    print_time = False
