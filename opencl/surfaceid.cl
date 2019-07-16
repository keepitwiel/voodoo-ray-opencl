__kernel void surface_id(
    __global int *width,
    __global int *height,
    __global float *position,
    __global float *direction,
    __global float *initial_propagation_length,
    __global uint *env_dim,
    __global ulong *environment,
    __global char *surface_rendered,
    __global char *intensity
) {
    /*
     * Given an initial 3d cartesian position and
     * 3d cartesian direction of a ray, calculate at which
     * surface it arrives after travelling through empty space
     */
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = *height * x + y;

    float3 p = vload3(i, position);
    float3 d = vload3(i, direction);
    float3 q = p;

    int3 n = {0, 0, 0};
    int3 qi = {0, 0, 0};

    uint3 dims = vload3(0, env_dim);

    char3 v = {
        (char) 0,
        (char) 0,
        (char) 0
    };

    float propagation_length = *initial_propagation_length;

    uint delta = 0;
    uint env_type = OPEN_SPACE;
    uint env_idx = 0;
    uint rendered_idx = 0;

    // find next non-empty block
    for (int k=0;k<200;k++) {
        //propagate
        q = p + d * propagation_length;
        delta = blocks_ahead(p, q);

        //decide what to do
        if (delta == 0) {
            // keep on trucking
            //printf("keep on trucking\n");
            p = q;
        }
        if (delta == 1) {
            // guaranteed found the next block with no others in between
            qi = convert_int3(q);
            n = convert_int3(p) - qi;

            if (within_bounds(qi, dims))
            {
                // we haven't left the environment
                env_idx = environment_index(qi, dims);
                //printf("%x\n", environment[env_idx]);
                if (environment[env_idx] > 0) {
                    env_type = (environment[env_idx] & 0xFF00000000000000) >> 56;
                    break;
                }
            } else {
                // we have left the environment
                env_type = GLOBAL_LIGHT;
                break;
            }
            // we're still in the environment and haven't hit a wall yet;
            // we need to keep on trucking
            p = q;
            propagation_length = *initial_propagation_length;
        }
        if (delta > 1) {
            // we have found a next block, but there might be one or more in between; reduce speed
            propagation_length *= PROPAGATION_REDUCTION_FACTOR;
        }
    }

    if (env_type == WALL || env_type == LOCAL_LIGHT) {
//        if (n.x ==  1) {v.x = 255; v.y = 0; v.z = 0;}
//        else if (n.x == -1) {v.x = 0; v.y = 63; v.z = 63;}
//        else if (n.y ==  1) {v.x = 0; v.y = 255; v.z = 0;}
//        else if (n.y == -1) {v.x = 63; v.y = 0; v.z = 63;}
//        else if (n.z ==  1) {v.x = 0; v.y = 0; v.z = 255;}
//        else if (n.z == -1) {v.x = 63; v.y = 63; v.z = 0;}
//        else {
//            printf("%d %d %d\n", n.x, n.y, n.z);
//            v.x = 0; v.y = 0; v.z = 0;
//        }
//
//        vstore3(v, i, intensity);

        int j = -1;
        if (n.x ==  1) j = 0;
        if (n.x == -1) j = 1;
        if (n.y ==  1) j = 2;
        if (n.y == -1) j = 3;
        if (n.z ==  1) j = 4;
        if (n.z == -1) j = 5;

        if (j >= 0) {
            rendered_idx = env_idx * 18 + j * 3;
            v.x = surface_rendered[rendered_idx];
            v.y = surface_rendered[rendered_idx + 1];
            v.z = surface_rendered[rendered_idx + 2];
            vstore3(v, i, intensity);
        }
    }
}
