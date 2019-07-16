int propagate(float3 *p, float3 *q, float3 *d,
              float3 *color, uint3 *env_prop,
              float initial_propagation_length,
              uint3 dims, __global ulong *environment) {
    /*
    Propagates a ray until it hits something that's not empty space,
    and returns a flag along with updated variables

    p: start position of ray
    q:
    */
    int flag = PROPAGATE;
    uint env_type = 0;
    uint env_idx = 0;
    int3 qi = {0, 0, 0};
    float3 col = {0.0f, 0.0f, 0.0f};
    ulong env = 0;

    while (flag == PROPAGATE) {
        next_block(p, q, d, initial_propagation_length);
        qi = convert_int3(*q);

        if (within_bounds(qi, dims))
        {
            env_idx = environment_index(qi, dims);
            env = environment[env_idx];

            env_type = extract_env_type(env); //(env & 0xFF00000000000000) >> 56;

            *env_prop = (uint3){
                (env & 0x00FF000000000000) >> 48, // 0 = pure diffusion, 0xFF = pure reflection
                (env & 0x0000FF0000000000) >> 40, // not implemented yet
                (env & 0x000000FF00000000) >> 32 // not implemented yet
            };

            *color = extract_color(env);

            //printf("%lu\n", env);
            if (env_type == OPEN_SPACE) *p = *q; // flag stays on PROPAGATE
            else if (env_type == WALL) flag = WALL;
            else if (env_type == LOCAL_LIGHT) flag = LOCAL_LIGHT;
            else if (env_type == MIRROR) flag = MIRROR;
            else flag = TERMINATE;
        } else {
            flag = GLOBAL_LIGHT;
        }
    }
    return flag;
}
