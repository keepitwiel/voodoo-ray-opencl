__kernel void calculate_ray_origins(
    __global int *width,
    __global int *height,
    __global float *position_in,
    __global float *direction_in,
    __global float *field_of_view,
    __global float *ray_spacing,
    __global int *camera_style,
    __global float *position_out,
    __global float *direction_out
) {
    /*
     * Given an initial 3d cartesian position and
     * 2d angular direction of the camera,
     * calculate for each ray originating from the
     * camera's pixel grid the 3d cartesian position and
     * 3d cartesian direction.
     */

    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint i = *height * x + y;

    float3 p = vload3(0, position_in);
    float2 d_angular = vload2(0, direction_in);

    float max_dimension = max(*width, *height);

    if (*camera_style == FISH_EYE) {
        // fish eye view
        float angular_spacing = *field_of_view / max_dimension;
        d_angular.x += x * angular_spacing - (*field_of_view / 2.0f) * (*width / max_dimension);
        d_angular.y -= y * angular_spacing + (*field_of_view / 2.0f) * (*height / max_dimension);
    } else if (*camera_style == FLAT_VIEW) {
        // "flat" view
        float triangle_side_length = tan(*field_of_view/2.0f) / max_dimension;
        float xx = x * triangle_side_length - 0.5f * (*width / max_dimension);
        float yy = y * triangle_side_length - 0.5f * (*height / max_dimension);
        d_angular.x += atan(xx);
        d_angular.y += atan(-yy);
    } else if (*camera_style == INFINITE) {
        // doesn't really work
        float xx = (x - 0.5f * (*width / max_dimension)) * *ray_spacing;
        float yy = (y - 0.5f * (*height / max_dimension)) * *ray_spacing;
    }

    float3 d = {
        cos(d_angular.x) * cos(d_angular.y),
        sin(d_angular.x) * cos(d_angular.y),
        sin(d_angular.y)
    };

    vstore3(p, i, position_out);
    vstore3(d, i, direction_out);
}