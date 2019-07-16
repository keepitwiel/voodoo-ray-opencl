__kernel void trace
(
    __global float *initial_propagation_length,
    __global int *width,
    __global int *height,
    __global float *position,
    __global float *direction,
    __global char *intensity,
    __global uint *env_dim,
    __global ulong *environment,
    __global int *seed
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = *height * x + y;

    // random variables
    long s = seed[i];
    float r1 = long2float(s), r2;
    // crappy way to generate new random variable
    s = rnd_long(s);
    r2 = long2float(s);
    seed[i] = s;

    // position/direction variables
    float3 p = vload3(i, position);
    float3 d = vload3(i, direction);
    int3 n = {0, 0, 0};
    uint3 dims = vload3(0, env_dim);
    //int camera_style = FLAT_VIEW;

    // ray variables
    float3 intensity_tmp = {
        MAX_INT_AS_FLOAT,
        MAX_INT_AS_FLOAT,
        MAX_INT_AS_FLOAT
    };

    //printf("%u... GO %2.3f ", i, intensity_tmp.x/MAX_INT_AS_FLOAT);

    float3 color = {0.0f, 0.0f, 0.0f};
    uint3 env_prop = {0, 0, 0};

    int flag = PROPAGATE;

    while (flag != TERMINATE) {
        if (flag == PROPAGATE) {
            flag = propagate_fast(
                &p, &d, &n, &color, &env_prop,
                *initial_propagation_length,
                dims, environment);
        } else if (flag == WALL) {
            intensity_tmp = intensity_tmp * color;
            if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                intensity_tmp *= 0.0f;
                flag = TERMINATE;
            } else {
                if ((s & 0x00000000000000FF) < env_prop.x) {
                    d = mirror2(n, d);
                } else {
                    r2 = r1;
                    s = rnd_long(s);
                    r1 = long2float(s);
                    seed[i] = s;
                    d = diffuse_angular2(n, r1, r2);
                }
                flag = PROPAGATE;
            }
        } else if (flag == MIRROR) {
            intensity_tmp = intensity_tmp * color;
            if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                intensity_tmp *= 0.0f;
                flag = TERMINATE;
            } else {
                d = mirror2(n, d);
                flag = PROPAGATE;
            }
        } else if (flag == LOCAL_LIGHT) {
            intensity_tmp = intensity_tmp * color;
            flag = TERMINATE;
        } else if (flag == GLOBAL_LIGHT) {
            intensity_tmp *= get_color_from_global_light(d);
            flag = TERMINATE;
        } else flag = TERMINATE;
    }

    char3 v = {
        (char)(((uint)(intensity_tmp.x) & 0xFF000000) >> 24),
        (char)(((uint)(intensity_tmp.y) & 0xFF000000) >> 24),
        (char)(((uint)(intensity_tmp.z) & 0xFF000000) >> 24)
    };

    vstore3(v, i, intensity);
}
