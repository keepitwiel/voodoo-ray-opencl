#define MAX_INT_AS_FLOAT 4294967295.0f
#define INTENSITY_THRESHOLD 1000000000.0f
#define MIN_PROPAGATION_LENGTH 0.001f
#define PROPAGATION_REDUCTION_FACTOR 0.3f
#define PI 3.141592654f
#define SQRT_OF_HALF 0.7071067811865475f
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


long rnd_long(long s) {
    return (s * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
}

float long2float(long s) {
    return 2.0f * (s & 0xffffffff) / MAX_INT_AS_FLOAT - 1.0f;
}

uint random_index_generator(uint seed1, uint seed2) {
    return (seed1 + seed2) % (1024 * 1024);
}

uint environment_index(int3 position, uint3 dimension) {
    return position.x * dimension.y * dimension.z + position.y * dimension.z + position.z;
}

int within_bounds(int3 position, uint3 dimension) {
    return (
        position.x >= 0 && position.y >= 0 && position.z >= 0 &&
        position.x < dimension.x && position.y < dimension.y && position.z < dimension.z
    );
}

uint extract_env_type(ulong env) {
    return (env & 0xFF00000000000000) >> 56;
}

float3 extract_color(ulong env) {
    float3 color = {
        ((env & 0x00000000FF000000) >> 24) / 255.0f,
        ((env & 0x0000000000FF0000) >> 16) / 255.0f,
        ((env & 0x000000000000FF00) >> 8) / 255.0f
    };

    return color;
}

float3 scale_float3(float3 u, float a) {
    u.x *= a;
    u.y *= a;
    u.z *= a;

    return u;
}

char3 intensity_to_rgb(float3 intensity) {
    char3 v = {
        (char)(((uint)(intensity.x) & 0xFF000000) >> 24),
        (char)(((uint)(intensity.y) & 0xFF000000) >> 24),
        (char)(((uint)(intensity.z) & 0xFF000000) >> 24)
    };

    return v;
}

float3 diffuse(float3 p, float3 q, float r1, float r2) {
    int3 n = convert_int3(p) - convert_int3(q);

    float3 d = {
        (n.x != 0) ? n.x * fabs(r1 * SQRT_OF_HALF) : r1 * SQRT_OF_HALF,
        (n.y != 0) ? n.y * fabs(r2 * SQRT_OF_HALF) : r2 * SQRT_OF_HALF,
        (n.z != 0) ? n.z * fabs(sqrt(1.0f - 0.5f * (r1*r1 + r2*r2))) : sqrt(1.0f - 0.5f * (r1*r1 + r2*r2))
    };

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

float3 diffuse_angular(int3 n, float r1, float r2) {
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

float3 diffuse_pseudorandom(int3 n, float3 r) {
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

float3 mirror2(int3 n, float3 d) {
    d.x = (n.x != 0) ? -d.x : d.x;
    d.y = (n.y != 0) ? -d.y : d.y;
    d.z = (n.z != 0) ? -d.z : d.z;

    return d;
}

float3 get_color_from_global_light(float3 d) {

    return (float3){1.0f, 1.0f, 1.0f};

//    float phi = acos(d.x);
//    float theta = asin(d.z);
//    if (phi > 0.4f*PI && phi < 0.45f*PI && theta > 0.15f*PI && theta < 0.2f*PI) {
//        return (float3){1.0f, 1.0f, 0.1f}; // yellow sun
//    } else {
//        return (float3){0.5f, 0.5f, 1.0f}; // blue sky
//        //return (float3){1.0f, 1.0f, 0.1f}; // yellow sky
//        //return (float3){0.0f, 0.0f, 0.0f}; // black sky
//    }
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
