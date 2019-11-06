#define MAX_INT_AS_FLOAT 4294967295.0f
#define INTENSITY_THRESHOLD 0.1f
#define MIN_PROPAGATION_LENGTH 0.001f
#define PROPAGATION_REDUCTION_FACTOR 0.3f
#define MAX_CUMULATIVE_LENGTH 200.0f
#define PI 3.141592654f

#define PROPAGATE -1
#define TERMINATE -2

#define OPEN_SPACE 0
#define WALL 1
#define LOCAL_LIGHT 2
#define GLOBAL_LIGHT 3
#define MIRROR 4

#define RADIUS 2

uint l1(int3 a) {
    return (uint)(abs(a.x) + abs(a.y) + abs(a.z));
}

float dist(float3 a, float3 b) {
    return sqrt(
        (a.x-b.x)*(a.x-b.x) +
        (a.y-b.y)*(a.y-b.y) +
        (a.z-b.z)*(a.z-b.z)
    );
}

long rnd_long(long s) {
    return (s * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
}

float long2float(long s) {
    return (s & 0xffffffff) / MAX_INT_AS_FLOAT;
}

float3 diffuse(float3 p, float3 q, float r1, float r2) {
    int3 n = convert_int3(p) - convert_int3(q);

    float phi = r1*2.0*PI;
    float theta = r2*0.5*PI;

    float3 d = {
        cos(theta) * cos(phi),
        cos(theta) * sin(phi),
        sin(theta)
    };

    d.x = (n.x != 0) ? n.x * fabs(d.x) : d.x;
    d.y = (n.y != 0) ? n.y * fabs(d.y) : d.y;
    d.z = (n.z != 0) ? n.z * d.z : d.z;

    return d;
}

float3 diffuse_angular(float3 p, float3 q, float r1, float r2) {
    int3 n = convert_int3(p) - convert_int3(q);

    float phi = r1 * PI;
    float theta = 0.5f * r2 * PI;

    float x = cos(theta) * cos(phi);
    float y = cos(theta) * sin(phi);
    float z = sin(theta);

    float3 d = {
        (n.x != 0) ? n.x * fabs(x) : x,
        (n.y != 0) ? n.y * fabs(y) : y,
        (n.z != 0) ? n.z * fabs(z) : z
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

float3 diffuse_smoke(float r1, float r2) {
    float phi = r1*2.0*PI;
    float theta = (r2-0.5)*PI;

    float3 d = {
        cos(theta) * cos(phi),
        cos(theta) * sin(phi),
        sin(theta)
    };

    return d;
}

float3 get_color_from_global_light(float3 d) {
    return (float3){0.5f, 0.5f, 1.0f};
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

__kernel void trace
(
    __global float *initial_propagation_length,
    __global int *width,
    __global int *height,
    __global float *position,
    __global float *direction,
    __global float *intensity,
    __global uint *env_dim,
    __global ulong *environment,
    __global int *seed,
    __global int *scale
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = *height * x + y;
    uint j = (*height * *scale) * (*scale * x) + (*scale * y);

    // random variables
    long s = seed[i];
    float r1 = long2float(s), r2;
    // crappy way to generate new random variable
    s = rnd_long(s);
    r2 = long2float(s);
    seed[i] = s;

    // position/direction variables
    float3 p = vload3(j, position);
    float3 d = vload3(j, direction);
    float3 q = p;
    uint3 dims = vload3(0, env_dim);

    // ray variables
    float3 intensity_tmp = {1.0f, 1.0f, 1.0f};
    float3 color = {0.0f, 0.0f, 0.0f};
    uint3 env_prop = {0, 0, 0};

    int flag = PROPAGATE;

    while (flag != TERMINATE) {
        if (flag == PROPAGATE) {
            flag = propagate(
                &p, &q, &d, &color, &env_prop,
                *initial_propagation_length,
                dims, environment, s, r1, r2);
        } else if (flag == WALL) {
            intensity_tmp = intensity_tmp * color;
            if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                intensity_tmp *= 0.0f;
                flag = TERMINATE;
            } else {
                if ((s & 0x00000000000000FF) < env_prop.x) {
                    d = mirror(p, q, d);
                } else {
                    r2 = r1;
                    s = rnd_long(s);
                    r1 = long2float(s);
                    seed[i] = s;
                    d = diffuse(p, q, r1, r2);
                }
                flag = PROPAGATE;
            }
        } else if (flag == MIRROR) {
            intensity_tmp = intensity_tmp * color;
            if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                intensity_tmp *= 0.0f;
                flag = TERMINATE;
            } else {
                d = mirror(p, q, d);
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

    vstore3(intensity_tmp, i, intensity);
}

__kernel void lidar
(
    __global int *width,
    __global int *height,
    __global float *pos_in,
    __global float *dir_in,
    __global float *pos_out,
    __global float *dir_out,
    __global float *field_of_view,
    __global float *initial_propagation_length,
    __global uint *env_dim,
    __global uint *environment,
    __global float *distance
)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = *height * x + y;

    // position/direction variables
    float3 p = vload3(0, pos_in);
    float3 q = p;
    float2 d_angular = vload2(0, dir_in);
    uint3 dims = vload3(0, env_dim);

    float max_dimension = max(*width, *height);
    float angular_spacing = *field_of_view / max_dimension;
    float phi = d_angular.x + x * angular_spacing - (*field_of_view / 2) * (*width / max_dimension);
    float theta = d_angular.y - y * angular_spacing + (*field_of_view / 2) * (*height / max_dimension);

    float3 d = {
        cos(phi) * cos(theta),
        sin(phi) * cos(theta),
        sin(theta)
    };

    float3 color = {0.0f, 0.0f, 0.0f};
    uint3 env_prop = {0, 0, 0};
    long s = 0L;
    float r1 = 0.0f;
    float r2 = 0.0f;

    int flag = propagate(
        &p, &q, &d, &color, &env_prop,
        *initial_propagation_length,
        dims, environment, s, r1, r2);

    vstore3(p, i, pos_out);
    vstore3(d, i, dir_out);
    distance[i] = dist(p, vload3(0, pos_in));
}

__kernel void distance_blur(
    __global int *width,
    __global int *height,
    __global float *position,
    __global float *intensity,
    __global float *blurred,
    __global uint *scale,
    __global float *blur_distance
) {
    /*
     * Blurs raytraced image so that nearby samples are blended together.
     */
    int x_d = get_global_id(0); // x coordinate relating to the distance map
    int y_d = get_global_id(1); // y
    int ref_idx = *height * x_d + y_d;
    int u = x_d / *scale; // x coordinate relating to the distance map
    int v = y_d / *scale; // y
    int u_max = *width / *scale;
    int v_max = *height / *scale;
    int i = 0;
    int j = 0;
    float3 s = {0.0f, 0.0f, 0.0f};
    float3 p = {0.0f, 0.0f, 0.0f};
    float3 p_reference = vload3(ref_idx, position);
    float euclidean_distance = 0.0f;
    float pixel_distance = 0.0f;
    float inverse_distance = 1.0f;
    float cumulative_weight = 0.0f;
    float3 cumulative_intensity = {0.0f, 0.0f, 0.0f};

    for (int uu = u-RADIUS; uu <= u+RADIUS; uu++) {
        for (int vv = v-RADIUS; vv <= v+RADIUS; vv++) {
            if (uu >= 0 && uu < u_max && vv >= 0 && vv < v_max) {
                i = v_max * uu + vv;
                s = vload3(i, intensity);
                j = *height * (uu * *scale) + (vv * *scale);
                p = vload3(j, position);
                euclidean_distance = dist(p, p_reference);
                pixel_distance = sqrt(
                    (float)(uu - u) * (float)(uu - u) + (float)(vv - v) * (float)(vv - v)
                );
                inverse_distance = 1.0f / (
                    *blur_distance * pixel_distance * euclidean_distance + 1.0
                );
                cumulative_weight += inverse_distance;
                cumulative_intensity += (s * inverse_distance);
            }
        }
    }
    vstore3(cumulative_intensity / cumulative_weight, ref_idx, blurred);
}
