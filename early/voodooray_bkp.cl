#define MAX_INT_AS_FLOAT 4294967295.0f
#define INTENSITY_THRESHOLD 1000000000.0f
#define MIN_PROPAGATION_LENGTH 0.001f
#define PROPAGATION_REDUCTION_FACTOR 0.3f
#define MAX_CUMULATIVE_LENGTH 200.0f
#define ALBEDO 0.5f
#define PI 3.141592654f
#define ATMOSPHERIC_DECAY 0.02f
#define SMOKE_THRESHOLD 0.9f

#define PROPAGATE -1
#define TERMINATE -2

#define OPEN_SPACE 0
#define WALL 1
#define LOCAL_LIGHT 2
#define GLOBAL_LIGHT 3
#define SMOKE 4

uint l1(int3 a) {
    return (uint)(abs(a.x) + abs(a.y) + abs(a.z));
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

float3 get_intensity_from_global_light(float3 d) {
    return (float3){0.0f, 0.0f, 0.0f};
}

uint blocks_ahead(float3 p, float3 q) {
    int3 pi = convert_int3(p);
    int3 qi = convert_int3(q);
    return abs(pi.x - qi.x) + abs(pi.y - qi.y) + abs(pi.z - qi.z);
}

int propagate(float3 *p, float3 *q, float3 *d,
              float3 *color,
              float initial_propagation_length,
              uint env_dim, __global uint *environment) {
    /*
    Propagates a ray until it hits something that's not empty space,
    and returns a flag along with updated variables
    */
    int flag = PROPAGATE;
    uint env_type = 0;
    uint env_idx = 0;
    int3 qi = {0, 0, 0};
    float3 col = {0.0f, 0.0f, 0.0f};
    uint env = 0;
    uint delta = 0;
    float speed = initial_propagation_length;

    while (flag == PROPAGATE) {
        speed = initial_propagation_length;
        // first, get the next block
        for (;;) {
            *q = *p + *d * speed;
            delta = blocks_ahead(*p, *q);
            if (delta == 0) *p = *q;
            else if (delta == 1) break;
            else speed *= PROPAGATION_REDUCTION_FACTOR;
            if (speed <= MIN_PROPAGATION_LENGTH) break;
        }

        // we found the next block
        qi = convert_int3(*q);

        if (qi.x >= 0 && qi.y >= 0 && qi.z >= 0 &&
            qi.x < env_dim && qi.y < env_dim && qi.z < env_dim) {
            env_idx = qi.x * env_dim*env_dim + qi.y * env_dim + qi.z;
            env = environment[env_idx];

            // maybe color assignment not needed yet?
            col.x = ((env & 0xFF000000) >> 24) / 255.0f;
            col.y = ((env & 0x00FF0000) >> 16) / 255.0f;
            col.z = ((env & 0x0000FF00) >> 8) / 255.0f;
            *color = col;

            env_type = (env & 0x000000FF);

            if (env_type == OPEN_SPACE) *p = *q; // flag stays on PROPAGATE
            else if (env_type == WALL) flag = WALL;
            else if (env_type == LOCAL_LIGHT) flag = LOCAL_LIGHT;
            else if (env_type == SMOKE) flag = SMOKE;
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
    __global float *position,
    __global float *direction,
    __global char *intensity,
    __global uint *env_dim,
    __global uint *environment,
    __global int *seed
)
{
    uint i = get_global_id(0);

    // random variables
    long s = seed[i];
    float r1 = long2float(s), r2;
    // crappy way to generate new random variable
    s = rnd_long(s);
    r2 = long2float(s);
    seed[i] = s;

    // position/direction variables
    float3 p = vload3(i, position);
    float3 q = p;
    float3 d = vload3(i, direction);

    // ray variables
    float3 intensity_tmp = {
        MAX_INT_AS_FLOAT,
        MAX_INT_AS_FLOAT,
        MAX_INT_AS_FLOAT
    };

    //printf("%u... GO %2.3f ", i, intensity_tmp.x/MAX_INT_AS_FLOAT);

    float3 color = {0.0f, 0.0f, 0.0f};

    int flag = PROPAGATE;

    while (flag != TERMINATE) {
        if (flag == PROPAGATE) {
            flag = propagate(
                &p, &q, &d, &color,
                *initial_propagation_length,
                *env_dim, environment);
        } else if (flag == WALL) {
            intensity_tmp = intensity_tmp * color;
            if (length(intensity_tmp) < INTENSITY_THRESHOLD) {
                flag = TERMINATE;
            } else {
                r2 = r1;
                s = rnd_long(s);
                r1 = long2float(s);
                seed[i] = s;
                d = diffuse(p, q, r1, r2);
                flag = PROPAGATE;
            }
        } else if (flag == LOCAL_LIGHT) {
            intensity_tmp = intensity_tmp * color;
            flag = TERMINATE;
        } else if (flag == GLOBAL_LIGHT) {
            intensity_tmp = get_intensity_from_global_light(d);
            flag = TERMINATE;
        } else if (flag == SMOKE) {
            flag = PROPAGATE;
            p = q;
            if (r1 < SMOKE_THRESHOLD) {
                //printf("SMOKE\n");
                r2 = r1;
                s = rnd_long(s);
                r1 = long2float(s);
                seed[i] = s;
                d = diffuse_smoke(r1, r2);
                intensity_tmp *= 0.99f;
                if (length(intensity_tmp) < INTENSITY_THRESHOLD) flag = TERMINATE;
                //else flag = PROPAGATE;
            }
        } else flag = TERMINATE;
    }

    //printf("T @ %2.3f.\n", intensity_tmp.x/MAX_INT_AS_FLOAT);
//    intensity[i] =
//        ((uint)(intensity_tmp.x) & 0xFF000000) +
//        (((uint)(intensity_tmp.y) & 0xFF000000) >> 8) +
//        (((uint)(intensity_tmp.z) & 0xFF000000) >> 16);
    char3 v = {(char)(intensity_tmp.x / 21677216), (char)(intensity_tmp.y / 21677216), (char)(intensity_tmp.y / 21677216)};
//    char3 v = {255, 127, 0};
    //printf("%u %u %u\n", v.x, v.y, v.z);
    vstore3(v, i, intensity);
}

float3 project
(
    float3 p,
    float3 d,
    float initial_propagation_length,
    uint env_dim,
    __global uint *environment,
    float *distance
)
{
    float3 p_candidate;
    float length = initial_propagation_length;
    *distance = length;

    while (1) {
        p_candidate = p + d * length;

        uint env_idx = (uint)p_candidate.x * env_dim*env_dim + (uint)p_candidate.y * env_dim + (uint)p_candidate.z;
        uint env = environment[env_idx];

        if (env > 0) {
            length *= PROPAGATION_REDUCTION_FACTOR;
            if (length <= MIN_PROPAGATION_LENGTH) break;
        }
        else {
            p = p_candidate;
            *distance += length;
        }
    }
    return p;
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

    float3 p0 = vload3(0, pos_in);
    float2 d0 = vload2(0, dir_in);

    float max_dimension = max(*width, *height);
    float angular_spacing = *field_of_view / max_dimension;
    float phi = d0.x + x * angular_spacing - (*field_of_view / 2) * (*width / max_dimension);
    float theta = d0.y - y * angular_spacing + (*field_of_view / 2) * (*height / max_dimension);

    float3 d1 = {
        cos(phi) * cos(theta),
        sin(phi) * cos(theta),
        sin(theta)
    };

    float l = 0;

    float3 p1 = p0; //project(p0, d1, *initial_propagation_length, *env_dim, environment, &l);
    distance[i] = l;

    vstore3(p0, i, pos_out);
    vstore3(d1, i, dir_out);
}