/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_common_color_utils.glsl"
#include "gpu_shader_math_vector_lib.glsl"

vec3 spowf3(vec3 a, float b) {
  return vec3(
    sign(a.x)*pow(abs(a.x), b),
    sign(a.y)*pow(abs(a.y), b),
    sign(a.z)*pow(abs(a.z), b)
  );
}

float spowf(float a, float b) {
  return sign(a)*pow(abs(a), b);
}

vec3 lin2log(vec3 rgb, int tf, float generic_log2_min_expo, float generic_log2_max_expo) {
  if (tf == 0) { // ACEScct
    rgb.x = rgb.x > 0.0078125f ? (log2(rgb.x) + 9.72f) / 17.52f : 10.5402377416545f * rgb.x + 0.0729055341958355f;
    rgb.y = rgb.y > 0.0078125f ? (log2(rgb.y) + 9.72f) / 17.52f : 10.5402377416545f * rgb.y + 0.0729055341958355f;
    rgb.z = rgb.z > 0.0078125f ? (log2(rgb.z) + 9.72f) / 17.52f : 10.5402377416545f * rgb.z + 0.0729055341958355f;
  } else if (tf == 1) { // Arri LogC3 EI 800
    rgb.x = rgb.x > 0.010591f ? 0.24719f * (log(5.555556f * rgb.x + 0.052272f) / log(10.0f)) + 0.385537f : 5.367655f * rgb.x + 0.092809f;
    rgb.y = rgb.y > 0.010591f ? 0.24719f * (log(5.555556f * rgb.y + 0.052272f) / log(10.0f)) + 0.385537f : 5.367655f * rgb.y + 0.092809f;
    rgb.z = rgb.z > 0.010591f ? 0.24719f * (log(5.555556f * rgb.z + 0.052272f) / log(10.0f)) + 0.385537f : 5.367655f * rgb.z + 0.092809f;
  } else if (tf == 2){  // Arri LogC 4
    const float a = (pow(2.0f, 18.0f) - 16.0f) / 117.45f;
    const float b = (1023.0f - 95.0f) / 1023.0f;
    const float c = 95.0f / 1023.f;
    const float s = (7.f * log(2.0f) * pow(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
    const float t = (pow(2.0f, 14.0f * ((-1.0f * c) / b) + 6.0f) - 64.0f) / a;

    rgb.x = rgb.x >= t ? ((log2(a * rgb.x + 64.f) - 6.f) / 14.f) * b + c : (rgb.x - t) / s;
    rgb.y = rgb.y >= t ? ((log2(a * rgb.y + 64.f) - 6.f) / 14.f) * b + c : (rgb.y - t) / s;
    rgb.z = rgb.z >= t ? ((log2(a * rgb.z + 64.f) - 6.f) / 14.f) * b + c : (rgb.z - t) / s;
  } else if (tf == 3) { // User controlled PureLog2
    rgb = log2(rgb / 0.18f);
    rgb = clamp(rgb, generic_log2_min_expo, generic_log2_max_expo);

    rgb = (rgb + abs(generic_log2_min_expo)) / (abs(generic_log2_min_expo)+abs(generic_log2_max_expo));
  }
  return rgb;
}

float sigmoid(float in_val, float sp, float tp, float Pslope, float px, float py, float s0, float t0)
{
  //calculate Shoulder;
  float ss =spowf(((spowf((Pslope*((s0-px)/(1-py))),sp)-1)*(spowf(Pslope*(s0-px),-sp))),-1/sp);
  float ms = Pslope*(in_val-px)/ss;
  float fs = ms/spowf(1+(spowf(ms,sp)),1/sp);

  //calculate Toe
  float ts =spowf(((spowf((Pslope*((px-t0)/(py))),tp)-1)*(spowf(Pslope*(px-t0),-tp))),-1/tp);
  float mr = (Pslope*(in_val-px))/-ts;
  float ft = mr/spowf(1+(spowf(mr,tp)),1/tp);

  in_val = in_val >= px ? ss * fs + py : (-ts * ft) + py;

  return in_val;
}

float lerp_chromaticity_angle(float h1, float h2, float t) {
    float delta = h2 - h1;
    if (delta > 0.5f) delta -= 1.0f;
    else if (delta < -0.5f) delta += 1.0f;
    float lerped = h1 + t * delta;
    return lerped - floor(lerped);
}

vec3 compensate_low_side(vec3 rgb, bool use_hacky_lerp, mat4 input_pri_to_rec2020_mat) {
    // Hardcoded Rec.2020 luminance coefficients (2015 CMFs)
    const vec3 luminance_coeffs = vec3(0.265818f, 0.59846986f, 0.1357121f);

    // Convert RGB to Rec.2020 for luminance calculation
    vec3 rgb_rec2020 = (input_pri_to_rec2020_mat * vec4(rgb, 1.0)).rgb;

    // Calculate original luminance Y
    float Y = rgb_rec2020.x * luminance_coeffs.x +
              rgb_rec2020.y * luminance_coeffs.y +
              rgb_rec2020.z * luminance_coeffs.z;

    // Calculate inverse RGB in working space
    float max_rgb = max(rgb.x, max(rgb.y, rgb.z));
    vec3 inverse_rgb = vec3(max_rgb - rgb.x, max_rgb - rgb.y, max_rgb - rgb.z);

    // Calculate max of the inverse
    float max_inv_rgb = max(inverse_rgb.x, max(inverse_rgb.y, inverse_rgb.z));

    // Convert inverse RGB to Rec.2020 for Y calculation
    vec3 inverse_rec2020 = (input_pri_to_rec2020_mat * vec4(inverse_rgb, 1.0)).rgb;
    float Y_inverse = inverse_rec2020.x * luminance_coeffs.x +
                      inverse_rec2020.y * luminance_coeffs.y +
                      inverse_rec2020.z * luminance_coeffs.z;

    // Calculate compensation values
    float y_compensate = (max_inv_rgb - Y_inverse + Y);
    if (use_hacky_lerp) {
        float Y_clipped = clamp(pow(Y, 0.08f), 0.0f, 1.0f);
        y_compensate = y_compensate + Y_clipped * (Y - y_compensate);
    }

    // Offset to avoid negatives
    float min_rgb = min(rgb.x, min(rgb.y, rgb.z));
    float offset = max(-min_rgb, 0.0f);
    vec3 rgb_offset = vec3(rgb.x + offset, rgb.y + offset, rgb.z + offset);

    // Calculate max of the offseted RGB
    float max_offset = max(rgb_offset.x, max(rgb_offset.y, rgb_offset.z));

    // Calculate new luminance after offset
    vec3 offset_rec2020 = (input_pri_to_rec2020_mat * vec4(rgb_offset, 1.0)).rgb;
    float Y_new = offset_rec2020.x * luminance_coeffs.x +
                  offset_rec2020.y * luminance_coeffs.y +
                  offset_rec2020.z * luminance_coeffs.z;

    // Calculate the inverted RGB offset
    vec3 inverse_offset = vec3(max_offset - rgb_offset.x,
                                       max_offset - rgb_offset.y,
                                       max_offset - rgb_offset.z);

    // Calculate max of the inverse
    float max_inv_offset = max(inverse_offset.x, max(inverse_offset.y, inverse_offset.z));

    vec3 inverse_offset_rec2020 = (input_pri_to_rec2020_mat * vec4(inverse_offset, 1.0)).rgb;
    float Y_inverse_offset = inverse_offset_rec2020.x * luminance_coeffs.x +
                             inverse_offset_rec2020.y * luminance_coeffs.y +
                             inverse_offset_rec2020.z * luminance_coeffs.z;

    float Y_new_compensate = (max_inv_offset - Y_inverse_offset + Y_new);
    if (use_hacky_lerp) {
        float Y_new_clipped = clamp(pow(Y_new, 0.08f), 0.0f, 1.0f);
        Y_new_compensate = Y_new_compensate + Y_new_clipped * (Y_new - Y_new_compensate);
    }

    // Adjust luminance ratio
    float ratio = (Y_new_compensate > y_compensate) ? (y_compensate / Y_new_compensate) : 1.0f;
    return vec3(rgb_offset.x * ratio, rgb_offset.y * ratio, rgb_offset.z * ratio);
}

void node_composite_agx_view_transform(vec4 color,
                                       float log2_min_in,
                                       float log2_max_in,
                                       float general_contrast_in,
                                       float toe_contrast_in,
                                       float shoulder_contrast_in,
                                       float pivot_offset_in,                     
                                       float per_channel_hue_flight_in,
                                       bool compensate_negatives_in,
                                       float p_working_log,
                                       mat4 scene_linear_to_working,
                                       mat4 working_to_display,
                                       mat4 display_to_scene_linear,
                                       float log_midgray,
                                       float midgray,
                                       mat4 insetmat,
                                       mat4 outsetmat,
                                       mat4 working_to_rec2020,
                                       mat4 display_to_rec2020,
                                       out vec4 result)
{
  result = color;
}
