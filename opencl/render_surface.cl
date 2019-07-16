

__kernel void render_surface(
    __global int *seed,
    __global float *initial_propagation_length,
    __global int *number_of_samples,
    __global uint *env_dim,
    __global ulong *environment,
    __global char *surface_rendered
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint z = get_global_id(2);
    uint3 dims = vload3(0, env_dim);

    // start block variables
    int3 start = {x, y, z};
    uint i = environment_index(start, dims);
    ulong env_start = environment[i];
    uint env_start_type = extract_env_type(env_start);

    // target block variables
    int3 target;// = start;
    ulong env_target;// = environment[environment_index(target, dims)];
    uint env_target_type;// = extract_env_type(env_target);

    // random variables
    long s = seed[i];
    float r1 = long2float(s), r2;
    // crappy way to generate new random variable
    s = rnd_long(s);
    r2 = long2float(s);
    seed[i] = s;

    int3 u[] = {
        { 1, 0, 0},
        {-1, 0, 0},
        { 0, 1, 0},
        { 0,-1, 0},
        { 0, 0, 1},
        { 0, 0,-1}
    };

    int j;

    float3 p_start = {0, 0, 0};
    float3 p_target = {0, 0, 0};
    float3 pp = {0, 0, 0};
    float3 qq = {0, 0, 0};

    float3 d = {0, 0, 0};

    float3 intensity_agg = {0, 0, 0};
    float3 intensity_tmp = {0, 0, 0};

    float3 color = {0.0f, 0.0f, 0.0f};
    uint3 env_prop = {0, 0, 0};
    int kmax = *number_of_samples;

    char3 v = {0, 0, 0};

    if (env_start_type != OPEN_SPACE) {

        for (j=0;j<6;j++) {
            // for each side, calculate start position of ray
            p_start.x = start.x + u[j].x * 0.5f + 0.5f;
            p_start.y = start.y + u[j].y * 0.5f + 0.5f;
            p_start.z = start.z + u[j].z * 0.5f + 0.5f;

            p_target.x = p_start.x + u[j].x;
            p_target.y = p_start.y + u[j].y;
            p_target.z = p_start.z + u[j].z;

            //see if target block in direction of ray is empty or not
            target = convert_int3(p_target);

            if (within_bounds(target, dims))
            {
                env_target = environment[environment_index(target, dims)];
                env_target_type = extract_env_type(env_target);

                if (env_target_type == OPEN_SPACE) {

                    } if (env_start_type == LOCAL_LIGHT) {
                        intensity_agg = extract_color(env_start);
                        intensity_agg = scale_float3(intensity_agg, MAX_INT_AS_FLOAT);

                    } else if (env_start_type == WALL) {
                        for (int k=0;k<kmax;k++) {
                            int flag = PROPAGATE;

                            intensity_tmp = extract_color(env_start);
                            intensity_tmp = scale_float3(intensity_tmp, MAX_INT_AS_FLOAT);

                            pp = p_start;
                            qq = p_start;

                            r2 = r1;
                            s = rnd_long(s);
                            r1 = long2float(s);
                            seed[i] = s;
                            d = diffuse_angular(p_start, p_target, r1, r2);

                            while (flag != TERMINATE) {
                                if (flag == PROPAGATE) {
                                    flag = propagate(
                                        &pp, &qq, &d, &color, &env_prop,
                                        *initial_propagation_length,
                                        dims, environment);
                                } else if (flag == WALL) {
                                    intensity_tmp = intensity_tmp * color;
                                    if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                                        intensity_tmp *= 0.0f;
                                        flag = TERMINATE;
                                    } else {

                                        r2 = r1;
                                        s = rnd_long(s);
                                        r1 = long2float(s);
                                        seed[i] = s;
                                        d = diffuse_angular(pp, qq, r1, r2);

                                        flag = PROPAGATE;
                                    }
                                } else if (flag == LOCAL_LIGHT) {
                                    //printf("local light | %d %d %d\n", u[j].x, u[j].y, u[j].z);
                                    intensity_tmp = intensity_tmp * color;
                                    flag = TERMINATE;
                                } else if (flag == GLOBAL_LIGHT) {
                                    //printf("global light | %d %d %d\n", u[j].x, u[j].y, u[j].z);
                                    intensity_tmp *= get_color_from_global_light(d);
                                    flag = TERMINATE;
                                } else flag = TERMINATE;
                            }

                            intensity_agg.x += intensity_tmp.x / kmax;
                            intensity_agg.y += intensity_tmp.y / kmax;
                            intensity_agg.z += intensity_tmp.z / kmax;
                        }
                }

                v = intensity_to_rgb(intensity_agg);
                vstore3(v, i * 6 + j, surface_rendered);

            }
        }
    }
}