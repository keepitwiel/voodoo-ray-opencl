#define MAX_INT_AS_FLOAT 4294967295.0f
#define INTENSITY_THRESHOLD 1000000000.0f
#define MIN_PROPAGATION_LENGTH 0.001f
#define PROPAGATION_REDUCTION_FACTOR 0.3f
#define MAX_CUMULATIVE_LENGTH 200.0f
#define PI 3.141592654f
#define NR_OF_SIDES 6
#define SIDE_DIRECTIONS
#define PROPAGATE -1
#define TERMINATE -2

#define OPEN_SPACE 0
#define WALL 1
#define LOCAL_LIGHT 2
#define GLOBAL_LIGHT 3
#define MIRROR 4

#define FISH_EYE 0
#define FLAT_VIEW 1
#define INFINITE 2

uint l1(int3 a) {
    return (uint)(abs(a.x) + abs(a.y) + abs(a.z));
}

long rnd_long(long s) {
    return (s * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
}

float long2float(long s) {
    return 2.0f * (s & 0xffffffff) / MAX_INT_AS_FLOAT - 1.0;
}

float3 diffuse(float3 p, float3 q, float r1, float r2) {
    int3 n = convert_int3(p) - convert_int3(q);

    float3 d = {
        (n.x != 0) ? n.x * fabs(r1) : r1,
        (n.y != 0) ? n.y * fabs(r2) : r2,
        (n.z != 0) ? n.z * fabs(sqrt(1.0f - r1*r1 - r2*r2)) : sqrt(1.0f - r1*r1 - r2*r2)
    };

    return d;
}


float3 diffuse_pseudorandom(float3 p, float3 q, float3 r) {
    int3 n = convert_int3(p) - convert_int3(q);

    float3 d = {
        (n.x != 0) ? n.x * fabs(r.x) : r.x,
        (n.y != 0) ? n.y * fabs(r.y) : r.y,
        (n.z != 0) ? n.z * fabs(r.z) : r.z
    };

    return d;
}


float3 mirror(float3 p, float3 q, float3 d) {
    int3 n = convert_int3(p) - convert_int3(q);

    d.x = (n.x != 0) ? -d.x : d.x;
    d.y = (n.y != 0) ? -d.y : d.y;
    d.z = (n.z != 0) ? -d.z : d.z;

    return d;
}

float3 get_color_from_global_light(float3 d) {
    float phi = acos(d.x);
    float theta = asin(d.z);
    if (phi > 0.4f*PI && phi < 0.45f*PI && theta > 0.15f*PI && theta < 0.2f*PI) {
        return (float3){1.0f, 1.0f, 0.1f};
    } else {
        return (float3){0.5f, 0.5f, 1.0f};
    }
}

uint blocks_ahead(float3 p, float3 q) {
    int3 pi = convert_int3(p);
    int3 qi = convert_int3(q);
    return abs(pi.x - qi.x) + abs(pi.y - qi.y) + abs(pi.z - qi.z);
}

void next_block(float3 *p, float3 *q, float3 *d, float propagation_length) {
    /*
     * given position p and direction d, propagate ray until it
     * hits the surface of the next block
     */
    uint delta = 0;
    for (;;) {
        //printf("prop length: %2.3f\n", propagation_length);
        *q = *p + *d * propagation_length;
        delta = blocks_ahead(*p, *q);
        if (delta == 0) *p = *q;
        else if (delta == 1) break;
        else propagation_length *= PROPAGATION_REDUCTION_FACTOR;
        if (propagation_length <= MIN_PROPAGATION_LENGTH) break;
    }
}

int propagate(float3 *p, float3 *q, float3 *d,
              float3 *color, uint3 *env_prop,
              float initial_propagation_length,
              uint3 dims, __global ulong *environment,
              long s, float r1, float r2) {
    /*
    Propagates a ray until it hits something that's not empty space,
    and returns a flag along with updated variables
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

        if (qi.x >= 0 && qi.y >= 0 && qi.z >= 0 &&
            qi.x < dims.x && qi.y < dims.y && qi.z < dims.z)
        {
            env_idx = qi.x * dims.y * dims.z + qi.y * dims.z + qi.z;
            env = environment[env_idx];

            env_type = (env & 0xFF00000000000000) >> 56;
            *env_prop = (uint3){
                (env & 0x00FF000000000000) >> 48, // 0 = pure diffusion, 0xFF = pure reflection
                (env & 0x0000FF0000000000) >> 40, // not implemented yet
                (env & 0x000000FF00000000) >> 32 // not implemented yet
            };

            *color = (float3){
                ((env & 0x00000000FF000000) >> 24) / 255.0f,
                ((env & 0x0000000000FF0000) >> 16) / 255.0f,
                ((env & 0x000000000000FF00) >> 8) / 255.0f
            };

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



__kernel void render_surface(
    __global float *random_vector,
    __global uint *random_index,
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
    uint ri = *random_index;
    float3 s2 = vload3(ri, random_vector);
    ulong s = 0;

    uint i = x * dims.y * dims.z + y * dims.z + z;

    float3 u[] = {
        { 1, 0, 0},
        {-1, 0, 0},
        { 0, 1, 0},
        { 0,-1, 0},
        { 0, 0, 1},
        { 0, 0,-1}
    };

    int j;

    uint env_idx = 0;

    float3 p = {0, 0, 0};
    float3 q = {0, 0, 0};
    float3 pp = {0, 0, 0};
    float3 qq = {0, 0, 0};

    float3 d = {0, 0, 0};
    int3 qi = {0, 0, 0};

    float3 intensity_agg = {0, 0, 0};
    float3 intensity_tmp = {0, 0, 0};

    float3 color = {0.0f, 0.0f, 0.0f};
    uint3 env_prop = {0, 0, 0};
    int kmax = *number_of_samples;

    //printf("%d %d %d | type: %lu\n", x, y, z, environment[i]);
    if (environment[i] > 0) {

        for (j=0;j<6;j++) {
            // for each side, calculate start position of ray
            p.x = x + u[j].x * 0.5f + 0.5f;
            p.y = y + u[j].y * 0.5f + 0.5f;
            p.z = z + u[j].z * 0.5f + 0.5f;

            printf("j, x, y, z: %d, %2.3f, %2.3f, %2.3f\n", j, p.x, p.y, p.z);

            //see if block in direction of ray is empty or not
            q = p + u[j];
            qi = convert_int3(q);

            if (qi.x >= 0 && qi.y >= 0 && qi.z >= 0 &&
                qi.x < dims.x && qi.y < dims.y && qi.z < dims.z)
            {
                env_idx = qi.x * dims.y * dims.z + qi.y * dims.z + qi.z;
                printf("position: %d %d %d %d | type: %lu\n", x, y, z, j, environment[env_idx]);
                if (environment[env_idx] == OPEN_SPACE) {

                    printf("open space\n");
                    for (int k=0;k<kmax;k++) {

                        int flag = PROPAGATE;

                        pp = p;
                        qq = p;

                        //TODO: need multiply by color of block at p. right now it's white light
                        intensity_tmp.x = MAX_INT_AS_FLOAT;
                        intensity_tmp.y = MAX_INT_AS_FLOAT;
                        intensity_tmp.z = MAX_INT_AS_FLOAT;


                        d = diffuse_pseudorandom(pp, qq, s2);
                        ri = (ri + 1) % (1024 * 1024);
                        random_index = ri;
                        s2 = vload3(ri, random_vector);


                        printf("random vector: %2.3f %2.3f %2.3f %d | index: %d\n", s2.x, s2.y, s2.z, j, ri);

//                        while (flag != TERMINATE) {
//                            if (flag == PROPAGATE) {
//                                flag = propagate(
//                                    &pp, &qq, &d, &color, &env_prop,
//                                    *initial_propagation_length,
//                                    dims, environment);
//                            } else if (flag == WALL) {
//                                intensity_tmp = intensity_tmp * color;
//                                if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
//                                    intensity_tmp *= 0.0f;
//                                    flag = TERMINATE;
//                                } else {
//                                    if ((s & 0x00000000000000FF) < env_prop.x) {
//                                        d = mirror(pp, qq, d);
//                                    } else {
//                                        d = diffuse_pseudorandom(pp, qq, s2);
//                                        ri = (ri + 1) % (1024 * 1024);
//                                        s2 = vload3(ri, random_vector);
//
//                                    }
//                                    flag = PROPAGATE;
//                                }
//                            } else if (flag == MIRROR) {
//                                intensity_tmp = intensity_tmp * color;
//                                if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
//                                    intensity_tmp *= 0.0f;
//                                    flag = TERMINATE;
//                                } else {
//                                    d = mirror(pp, qq, u[j]);
//                                    flag = PROPAGATE;
//                                }
//                            } else if (flag == LOCAL_LIGHT) {
//                                intensity_tmp = intensity_tmp * color;
//                                flag = TERMINATE;
//                            } else if (flag == GLOBAL_LIGHT) {
//                                intensity_tmp *= get_color_from_global_light(d);
//                                flag = TERMINATE;
//                            } else flag = TERMINATE;
//                        }



                        intensity_agg.x += intensity_tmp.x / kmax;
                        intensity_agg.y += intensity_tmp.y / kmax;
                        intensity_agg.z += intensity_tmp.z / kmax;

//                        printf("x y z j k %d %d %d %d %d | color: %2.3f %2.3f %2.3f\n",
//                            x, y, z, j, k,
//                            intensity_tmp.x / MAX_INT_AS_FLOAT, intensity_tmp.y / MAX_INT_AS_FLOAT, intensity_tmp.z / MAX_INT_AS_FLOAT
//                        );

                    }




                    char3 v = {
                        (char)(((uint)(intensity_agg.x) & 0xFF000000) >> 24),
                        (char)(((uint)(intensity_agg.y) & 0xFF000000) >> 24),
                        (char)(((uint)(intensity_agg.z) & 0xFF000000) >> 24)
                    };

                    vstore3(v, i * 6 + j, surface_rendered);
                }
            }
        }
    }
}