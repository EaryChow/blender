/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_common_color_utils.glsl"
#include "gpu_shader_math_vector_lib.glsl"

float4 spowf4(float4 a, float b) {
  return float4(
    sign(a.x)*pow(abs(a.x), b),
    sign(a.y)*pow(abs(a.y), b),
    sign(a.z)*pow(abs(a.z), b),
    a.w
  );
}

float spowf(float a, float b) {
  return sign(a)*pow(abs(a), b);
}

float4 lin2log(float4 rgba, int tf, float generic_log2_min_expo, float generic_log2_max_expo) {
  float3 rgb = rgba.rgb;
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
  return float4(rgb, rgba.a);
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

float4 compensate_low_side(float4 rgba, bool use_hacky_lerp, float4x4 input_pri_to_rec2020_mat) {
  float3 rgb = rgba.rgb;
    // Hardcoded Rec.2020 luminance coefficients (2015 CMFs)
    const float3 luminance_coeffs = float3(0.265818f, 0.59846986f, 0.1357121f);

    // Convert RGB to Rec.2020 for luminance calculation
    float3 rgb_rec2020 = (input_pri_to_rec2020_mat * float4(rgb, 1.0)).rgb;

    // Calculate original luminance Y
    float Y = rgb_rec2020.x * luminance_coeffs.x +
              rgb_rec2020.y * luminance_coeffs.y +
              rgb_rec2020.z * luminance_coeffs.z;

    // Calculate inverse RGB in working space
    float max_rgb = max(rgb.x, max(rgb.y, rgb.z));
    float3 inverse_rgb = float3(max_rgb - rgb.x, max_rgb - rgb.y, max_rgb - rgb.z);

    // Calculate max of the inverse
    float max_inv_rgb = max(inverse_rgb.x, max(inverse_rgb.y, inverse_rgb.z));

    // Convert inverse RGB to Rec.2020 for Y calculation
    float3 inverse_rec2020 = (input_pri_to_rec2020_mat * float4(inverse_rgb, 1.0)).rgb;
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
    float3 rgb_offset = float3(rgb.x + offset, rgb.y + offset, rgb.z + offset);

    // Calculate max of the offseted RGB
    float max_offset = max(rgb_offset.x, max(rgb_offset.y, rgb_offset.z));

    // Calculate new luminance after offset
    float3 offset_rec2020 = (input_pri_to_rec2020_mat * float4(rgb_offset, 1.0)).rgb;
    float Y_new = offset_rec2020.x * luminance_coeffs.x +
                  offset_rec2020.y * luminance_coeffs.y +
                  offset_rec2020.z * luminance_coeffs.z;

    // Calculate the inverted RGB offset
    float3 inverse_offset = float3(max_offset - rgb_offset.x,
                                       max_offset - rgb_offset.y,
                                       max_offset - rgb_offset.z);

    // Calculate max of the inverse
    float max_inv_offset = max(inverse_offset.x, max(inverse_offset.y, inverse_offset.z));

    float3 inverse_offset_rec2020 = (input_pri_to_rec2020_mat * float4(inverse_offset, 1.0)).rgb;
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
    return float4(rgb_offset.x * ratio, rgb_offset.y * ratio, rgb_offset.z * ratio, rgba.a);
}

void node_composite_agx_view_transform(float4 color,
                                       float log2_min_in,
                                       float log2_max_in,
                                       float general_contrast_in,
                                       float toe_contrast_in,
                                       float shoulder_contrast_in,
                                       float pivot_offset_in,
                                       float3 hue_flights_in,
                                       float3 rates_of_attenuation_in,
                                       float3 reverse_hue_flights_in,
                                       float3 restore_purity_in,
                                       float per_channel_hue_flight_in,
                                       float tinting_scale_in,
                                       float tinting_hue_in,
                                       float compensate_negatives_in,
                                       float p_working_log,
                                       float4x4 scene_linear_to_working,
                                       float4x4 working_to_display,
                                       float4x4 display_to_scene_linear,
                                       float log_midgray,
                                       float midgray,
                                       float4x4 insetmat,
                                       float4x4 outsetmat,
                                       float4x4 working_to_rec2020,
                                       float4x4 display_to_rec2020,
                                       out float4 result)
{

  color = scene_linear_to_working * color;

  // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
  if (bool(compensate_negatives_in)) {
    color = compensate_low_side(color, false, working_to_rec2020);
  }
  else {
    color = max(float4(0.0), color);
  }
  // apply inset matrix
  color = insetmat * color;
  // record pre-formation chromaticity angle
  float4 pre_curve_hsv;
  rgb_to_hsv(color, pre_curve_hsv);

  // encode to working log
  color = lin2log(color, int(p_working_log), log2_min_in, log2_max_in);

  // apply sigmoid, the image is formed at this point
  color.x = sigmoid(color.x, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray, 1.0f, 0.0f);
  color.y = sigmoid(color.y, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray, 1.0f, 0.0f);
  color.z = sigmoid(color.z, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray, 1.0f, 0.0f);
  float4 img = color;
  // Linearize the formed image assuming its native transfer function is Rec.1886 curve
  img = spowf4(img, 2.4f);

  // lerp pre- and post-curve chromaticity angle
  float4 post_curve_hsv;
  rgb_to_hsv(img, post_curve_hsv);
  post_curve_hsv[0] = lerp_chromaticity_angle(pre_curve_hsv[0], post_curve_hsv[0], per_channel_hue_flight_in);
  hsv_to_rgb(post_curve_hsv, img);

  // apply outset matrix
  img = outsetmat * img;

  // convert from working primaries to target display primaries
  img = working_to_display * img;

  // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
  if (bool(compensate_negatives_in))  {
    img = compensate_low_side(img, true, display_to_rec2020);
  }
  else {
    img = max(float4(0.0), img);
  }

  // convert linearized formed image back to OCIO's scene_linear role space
  img = display_to_scene_linear * img;

  result = img;
}