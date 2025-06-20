/* SPDX-FileCopyrightText: 2025 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "gpu_shader_common_color_utils.glsl"
#include "gpu_shader_math_vector_lib.glsl"

void node_composite_agx_view_transform(float4 color,
                                       float log2_min_in,
                                       float log2_max_in,
                                       float general_contrast_in,
                                       float toe_contrast_in,
                                       float shoulder_contrast_in,
                                       float pivot_offset_in,                     
                                       float per_channel_hue_flight_in,
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
  result = color;
}
