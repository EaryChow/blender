/* SPDX-FileCopyrightText: 2025 Blender Authors */
/* SPDX-License-Identifier: GPL-2.0-or-later */

// Include Headers
#include "BLI_math_base.hh"
#include "BLI_math_color.h"
#include "BLI_math_color.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector.h"
#include "BLI_math_vector_types.hh"
#include "BLI_string.h"
#include "BLT_translation.hh"
#include "BKE_node.hh"
#include "COM_node_operation.hh"
#include "DNA_node_types.h"
#include "GPU_material.hh"
#include "FN_multi_function_builder.hh"
#include "IMB_colormanagement.hh"
#include "NOD_multi_function.hh"
#include "NOD_node_declaration.hh"
#include "NOD_rna_define.hh"
#include "node_composite_util.hh"
#include "node_cmp_agx_utils.hh"
#include "RNA_access.hh"
#include "RNA_define.hh"
#include "RNA_enum_types.hh"
#include "UI_interface.hh"
#include "UI_resources.hh"


// Namespace Declaration
using namespace blender::compositor;

namespace blender::nodes::node_composite_agx_view_transform_cc {

// define enums
enum class AGXPrimaries : int16_t {
  AGX_PRIMARIES_AP0 = 0,
  AGX_PRIMARIES_AP1 = 1,
  AGX_PRIMARIES_P3D65 = 2,
  AGX_PRIMARIES_REC709 = 3,
  AGX_PRIMARIES_REC2020 = 4,
  AGX_PRIMARIES_AWG3 = 5,
  AGX_PRIMARIES_AWG4 = 6,
  AGX_PRIMARIES_EGAMUT = 7,
};

enum class AGXWorkingLog : int16_t {
  AGX_WORKING_LOG_ACESCCT = 0,
  AGX_WORKING_LOG_ARRI_LOGC3 = 1,
  AGX_WORKING_LOG_ARRI_LOGC4 = 2,
  AGX_WORKING_LOG_GENERIC_LOG2 = 3,
};

static const EnumPropertyItem agx_working_primaries_items[] = {
    {int(AGXPrimaries::AGX_PRIMARIES_AP0), "ap0", 0, "AP0", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AP1), "ap1", 0, "AP1", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_P3D65), "p3d65", 0, "P3-D65", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_REC709), "rec709", 0, "Rec.709", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_REC2020), "rec2020", 0, "Rec.2020", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AWG3), "awg3", 0, "ARRI Alexa Wide Gamut 3", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AWG4), "awg4", 0, "ARRI Alexa Wide Gamut 4", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_EGAMUT), "egamut", 0, "FilmLight E-Gamut", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem agx_display_primaries_items[] = {
  {int(AGXPrimaries::AGX_PRIMARIES_P3D65), "p3d65", 0, "P3-D65", ""},
  {int(AGXPrimaries::AGX_PRIMARIES_REC709), "rec709", 0, "Rec.709", ""},
  {int(AGXPrimaries::AGX_PRIMARIES_REC2020), "rec2020", 0, "Rec.2020", ""},
  {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem agx_working_log_items[] = {
    {int(AGXWorkingLog::AGX_WORKING_LOG_ACESCCT), "acescct", 0, "ACEScct", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_ARRI_LOGC3), "arri_logc3", 0, "ARRI LogC3", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_ARRI_LOGC4), "arri_logc4", 0, "ARRI LogC4", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2), "generic_log2", 0, "Generic Log2", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

// RNA functions for node properties
static void node_rna(StructRNA *srna) {
  PropertyRNA *prop;

  prop = RNA_def_node_enum(
      srna,
      "working_primaries",
      "Working",
      "The working primaries that the AgX mechanism applies to",
      agx_working_primaries_items, 
      NOD_inline_enum_accessors(custom2),
      int(AGXPrimaries::AGX_PRIMARIES_REC2020)); 

  prop = RNA_def_node_enum(
      srna,
      "working_log",
      "Log",
      "The Log curve applied before the sigmoid in the AgX mechanism",
      agx_working_log_items,
      NOD_inline_enum_accessors(custom3),
      int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2)); 

  prop = RNA_def_node_enum(
      srna,
      "display_primaries",
      "Display",
      "The primaries of the target display device",
      agx_display_primaries_items,
      NOD_inline_enum_accessors(custom4),
      int(AGXPrimaries::AGX_PRIMARIES_REC709));

  prop = RNA_def_node_boolean(
      srna,
      "sync_outset_to_inset",
      "Use for Restoration",
      "Use the same settings as Attenuation section for Purity Restoration, for ease of use",
      NOD_inline_boolean_accessors(custom1, 1),
      false);
}

// initialize
static void node_init(bNodeTree * /*tree*/, bNode *node) {
  node->custom2 =  int(AGXPrimaries::AGX_PRIMARIES_REC2020);
  node->custom3 = int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2);
  node->custom4 = int(AGXPrimaries::AGX_PRIMARIES_REC709);
  node->custom1 = false;
}

// Node Declaration
static void node_declare(NodeDeclarationBuilder &b) {
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  b.add_output<decl::Color>("Color");

  b.add_input<decl::Color>("Color")
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);

  /* Panel for log and sigmoid curve settings. */
  PanelDeclarationBuilder &curve_panel = b.add_panel("Curve").default_closed(false);

  curve_panel.add_input<decl::Float>("General Contrast")
    .default_value(2.4f)
    .min(1.4f)
    .max(4.0f)
    .subtype(PROP_FACTOR)
    .description(
        "Slope of the S curve."
        "Slope of the S curve. Controls the general contrast across the image");

  curve_panel.add_input<decl::Float>("Toe Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_FACTOR)
    .description(
        "Toe exponential power of the S curve."
        "Higher values make darker regions crush harder towards black");

  curve_panel.add_input<decl::Float>("Shoulder Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_FACTOR)
    .description(
        "Shoulder exponential power of the S curve."
        "Higher values make brighter regions crush harder towards white");

  curve_panel.add_input<decl::Float>("Contrast Pivot Offset")
    .default_value(0.0f)
    .min(-0.3f)
    .max(0.18f)
    .subtype(PROP_FACTOR)
    .short_label("Pivot Offset")
    .description(
        "Controls the pivot point for all contrast adjustments");

  curve_panel.add_layout([](uiLayout *layout, bContext * /*C*/, PointerRNA *ptr) {
    layout->prop(ptr, "working_log", UI_ITEM_R_SPLIT_EMPTY_NAME, std::nullopt, ICON_NONE);});

  curve_panel.add_input<decl::Float>("Log2 Minimum Exposure")
    .default_value(-10.0f)
    .min(-15.0f)
    .max(-5.0f)
    .subtype(PROP_NONE)
    .short_label("Log2 Min")
    .description(
        "The lower end of the generic log2 curve. Values are in Exposure stops."
        "Only in use when working log is set to Generic Log2");

  curve_panel.add_input<decl::Float>("Log2 Maximum Exposure")
    .default_value(6.5f)
    .min(2.8f)
    .max(38.0f)
    .subtype(PROP_NONE)
    .short_label("Log2 Max")
    .description(
        "The upper end of the log curve. Values are in Exposure stops."
        "Only in use when working log is set to Generic Log2");

  /* Panel for inset matrix settings. */
  PanelDeclarationBuilder &inset_panel = b.add_panel("Attenuation").default_closed(true);

  inset_panel.add_input<decl::Vector>("Hue Flights")
    .default_value({2.13976f, -1.22827f, -3.05174f})
    .min(-10.0f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Hue Rotation angle in degrees for each of the RGB primaries before curve."
        "Negative is clockwise, and positive is counterclockwise");

  inset_panel.add_input<decl::Vector>("Attenuation Rates")
    .default_value({0.329652f, 0.280513f, 0.124754f})
    .min(0.0f)
    .max(0.6f)
    .subtype(PROP_NONE)
    .description(
        "Percentage relative to the primary chromaticity purity,"
        "by which the chromaticity scales inwards before curve");

  inset_panel.add_layout([](uiLayout *layout, bContext * /*C*/, PointerRNA *ptr) {
    layout->prop(ptr, "sync_outset_to_inset", UI_ITEM_R_SPLIT_EMPTY_NAME, std::nullopt, ICON_NONE);});

  /* Panel for outset matrix settings. */
  PanelDeclarationBuilder &outset_panel = b.add_panel("Purity Restoration").default_closed(true);

  outset_panel.add_input<decl::Vector>("Reverse Hue Flights")
    .default_value({0.0f, 0.0f, 0.0f})
    .min(-10.0f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Hue Rotation angle in degrees for each of the RGB primaries after curve."
        "Direction is the reverse of the Attenuation. Negative is counterclockwise, positive is clockwise.");

  outset_panel.add_input<decl::Vector>("Restore Purity")
    .default_value({0.323174f, 0.283256f, 0.037433f})
    .min(0.0f)
    .max(0.6f)
    .subtype(PROP_NONE)
    .description(
        "Percentage relative to the primary chromaticity purity,"
        "by which the chromaticity scales outwards after curve");

  /* Panel for look adjustments settings. */
  PanelDeclarationBuilder &look_panel = b.add_panel("Look").default_closed(false);

  look_panel.add_input<decl::Float>("Per-Channel Hue Flight")
    .default_value(0.4f)
    .min(0.0f)
    .max(1.0f)
    .subtype(PROP_FACTOR)
    .description(
        "The percentage of hue shift introduced by the per-channel curve."
        "Higher value will have yellower orange, for example");

  look_panel.add_input<decl::Float>("Tinting Scale")
    .default_value(0.0f)
    .min(-0.2f)
    .max(0.2f)
    .subtype(PROP_FACTOR)
    .description(
        "Controls how far the white point shifts in the outset."
        "Affecting the intensity or strength of the tint applied after curve");

  look_panel.add_input<decl::Float>("Tinting Hue")
    .default_value(0.0f)
    .min(-180.0f)
    .max(180.0f)
    .subtype(PROP_FACTOR)
    .description(
        "Adjusts the direction in which the white point shifts in the outset."
        "influencing the overall hue of the tint applied after curve");

  /* Panel for primaries settings. */
  PanelDeclarationBuilder &primaries_panel = b.add_panel("Primaries").default_closed(true);

  primaries_panel.add_layout([](uiLayout *layout, bContext * /*C*/, PointerRNA *ptr) {
    layout->prop(ptr, "working_primaries", UI_ITEM_R_SPLIT_EMPTY_NAME, std::nullopt, ICON_NONE);
    layout->prop(ptr, "display_primaries", UI_ITEM_R_SPLIT_EMPTY_NAME, std::nullopt, ICON_NONE);
  });

  primaries_panel.add_input<decl::Bool>("Compensate for the Negatives")
    .default_value(true)
    .short_label("Compensate Negatives")
    .description(
        "Use special luminance compensation technique to prevent out-of-gamut negative values."
        "Done in both pre-curve and post-curve state.");

}

static void node_update(bNodeTree *ntree, bNode *node)
{
  // Get the value of the boolean property
  bool use_same_settings = node->custom1;
  bool outset_panel_sockets_available = !use_same_settings;
  // Find and set the availability of each related socket
  bNodeSocket *reverse_hue_flights = blender::bke::node_find_socket(*node, SOCK_IN, "Reverse Hue Flights");
  if (reverse_hue_flights) {
    blender::bke::node_set_socket_availability(*ntree, *reverse_hue_flights, outset_panel_sockets_available);
  }

  bNodeSocket *restore_purity = blender::bke::node_find_socket(*node, SOCK_IN, "Restore Purity");
  if (restore_purity) {
    blender::bke::node_set_socket_availability(*ntree, *restore_purity, outset_panel_sockets_available);
  }

  // Get the value of the enum property
  bool log2_settings_available = node->custom3 == int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2);
  // Find and set the availability of each related socket
  bNodeSocket *log2_exposure_min = blender::bke::node_find_socket(*node, SOCK_IN, "Log2 Minimum Exposure");
  if (log2_exposure_min) {
     blender::bke::node_set_socket_availability(*ntree, *log2_exposure_min, log2_settings_available);
  }

  bNodeSocket *log2_exposure_max = blender::bke::node_find_socket(*node, SOCK_IN, "Log2 Maximum Exposure");
  if (log2_exposure_max) {
    blender::bke::node_set_socket_availability(*ntree, *log2_exposure_max, log2_settings_available);
  }

}

// CPU processing logic
static float4 agx_image_formation(float4 color,
                                  float general_contrast_in,
                                  float toe_contrast_in,
                                  float shoulder_contrast_in,
                                  float pivot_offset_in,                     
                                  float log2_min_in,
                                  float log2_max_in,
                                  float3 hue_flights_in,
                                  float3 attenuation_rates_in,
                                  float3 reverse_hue_flights_in,
                                  float3 restore_purity_in,
                                  float per_channel_hue_flight_in,
                                  float tinting_scale_in,
                                  float tinting_hue_in,
                                  bool compensate_negatives_in,
                                  int p_working_primaries,
                                  int p_working_log,
                                  int p_display_primaries,
                                  bool p_use_inverse_inset
                                )
{
  float alpha = color.w;
  float in_rgb_array[3] = {color.x, color.y, color.z};
  float in_xyz_array[3];
  IMB_colormanagement_scene_linear_to_xyz(in_xyz_array, in_rgb_array);
  float3 in_xyz = make_float3(in_xyz_array[0], in_xyz_array[1], in_xyz_array[2]);

  float3x3 xyz_to_working = XYZtoRGB(COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);
  float3 rgb = mult_f3_f33(in_xyz, xyz_to_working);

  // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
  if (compensate_negatives_in) {
    rgb = compensate_low_side(rgb, false, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);
  }
  else {
    rgb = maxf3(0, rgb);
  }

  // generate inset matrix
  Chromaticities inset_chromaticities = InsetPrimaries(
      COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
      attenuation_rates_in.x, attenuation_rates_in.y, attenuation_rates_in.z,
      hue_flights_in.x, hue_flights_in.y, hue_flights_in.z);

  float3x3 insetmat = RGBtoRGB(inset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);

  // apply inset matrix
  rgb = mult_f3_f33(rgb, insetmat);

  // record pre-formation chromaticity angle
  float3 pre_curve_hsv;
  rgb_to_hsv_v(rgb, pre_curve_hsv);

  // encode to working log
  rgb = lin2log(rgb, static_cast<int>(p_working_log), log2_min_in, log2_max_in);

  // apply sigmoid, the image is formed at this point
  float log_midgray = lin2log(make_float3(0.18f, 0.18f, 0.18f), static_cast<int>(p_working_log), log2_min_in, log2_max_in).x;
  float image_native_power = 2.4f;
  float midgray = pow(0.18f, 1.0f / image_native_power);
  rgb.x = sigmoid(rgb.x, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray);
  rgb.y = sigmoid(rgb.y, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray);
  rgb.z = sigmoid(rgb.z, shoulder_contrast_in, toe_contrast_in, general_contrast_in, log_midgray + pivot_offset_in, midgray);
  float3 img = rgb;
  // Linearize the formed image assuming its native transfer function is Rec.1886 curve
  img = spowf3(img, image_native_power);

  // lerp pre- and post-curve chromaticity angle
  float3 post_curve_hsv;
  rgb_to_hsv_v(img, post_curve_hsv);
  post_curve_hsv[0] = lerp_chromaticity_angle(pre_curve_hsv[0], post_curve_hsv[0], per_channel_hue_flight_in);
  hsv_to_rgb_v(post_curve_hsv, img);

  // generate outset matrix
  float3x3 outsetmat;
  if (p_use_inverse_inset) {
    Chromaticities outset_chromaticities = InsetPrimaries(
        COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
        attenuation_rates_in.x, attenuation_rates_in.y, attenuation_rates_in.z, // Uses attenuation settings
        hue_flights_in.x, hue_flights_in.y, hue_flights_in.z,         // Uses attenuation settings
        tinting_hue_in + 180, tinting_scale_in);
    outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
  }
  else {
    Chromaticities outset_chromaticities = InsetPrimaries(
        COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
        restore_purity_in.x, restore_purity_in.y, restore_purity_in.z,
        reverse_hue_flights_in.x, reverse_hue_flights_in.y, reverse_hue_flights_in.z,
        tinting_hue_in + 180, tinting_scale_in);
    outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
  }

  // apply outset matrix
  img = mult_f3_f33(img, outsetmat);

  // convert from working primaries to target display primaries
  float3x3 working_to_display = RGBtoRGB(COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
                                         COLOR_SPACE_PRI[static_cast<int>(p_display_primaries)]);
  img = mult_f3_f33(img, working_to_display);

  // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
  if (compensate_negatives_in) {
    img = compensate_low_side(img, true, COLOR_SPACE_PRI[static_cast<int>(p_display_primaries)]);
  }
  else {
    img = maxf3(0, img);
  }

  // convert linearized formed image back to OCIO's scene_linear role space
  float3x3 display_to_xyz = RGBtoXYZ(COLOR_SPACE_PRI[static_cast<int>(p_display_primaries)]);
  float3 out_xyz = mult_f3_f33(img, display_to_xyz);
  float out_xyz_array[3] = {out_xyz.x, out_xyz.y, out_xyz.z};
  float img_array[3];
  IMB_colormanagement_xyz_to_scene_linear(img_array, out_xyz_array);

  // re-combine the alpha channel
  color.x = img_array[0];
  color.y = img_array[1];
  color.z = img_array[2];
  color.w = alpha;

  return color;
}

// GPU Processing
static int node_gpu_material(GPUMaterial *material,
  bNode *node,
  bNodeExecData * /*execdata*/,
  GPUNodeStack *inputs,
  GPUNodeStack *outputs)
{
return GPU_stack_link(material, node, "node_composite_agx_view_transform", inputs, outputs);
}

// Multi Function
static void node_build_multi_function(blender::nodes::NodeMultiFunctionBuilder &builder)
{
  const bool use_inverse_inset = builder.node().custom1;
  const bool use_generic_log2 = builder.node().custom3 == int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2);

  if (!use_inverse_inset && use_generic_log2) {
    builder.construct_and_set_matching_fn_cb([&]() {
      return mf::build::detail::build_multi_function_with_n_inputs_one_output<float4>(
          "AgX View Transform",
          [=, &builder](const float4 &color,
                       const float general_contrast_in,
                       const float toe_contrast_in,
                       const float shoulder_contrast_in,
                       const float pivot_offset_in,
                       const float log2_min_in,
                       const float log2_max_in,
                       const float3 hue_flights_in,
                       const float3 attenuation_rates_in,
                       const float3 reverse_hue_flights_in,
                       const float3 restore_purity_in,
                       const float per_channel_hue_flight_in,
                       const float tinting_scale_in,
                       const float tinting_hue_in,
                       const bool compensate_negatives_in) -> float4 {
            return agx_image_formation(
                color,
                general_contrast_in,
                toe_contrast_in,
                shoulder_contrast_in,
                pivot_offset_in,
                log2_min_in,
                log2_max_in,
                hue_flights_in,
                attenuation_rates_in,
                reverse_hue_flights_in,
                restore_purity_in,
                per_channel_hue_flight_in,
                tinting_scale_in,
                tinting_hue_in,
                compensate_negatives_in,
                builder.node().custom2,
                builder.node().custom3,
                builder.node().custom4,
                builder.node().custom1);
          },
          mf::build::exec_presets::SomeSpanOrSingle<0>(),
          TypeSequence<float4,
                       float,
                       float,
                       float,
                       float,
                       float,
                       float,
                       float3,
                       float3,
                       float3,
                       float3,
                       float,
                       float,
                       float,
                       bool>());
    });
  } else if (use_inverse_inset && use_generic_log2) {
    builder.construct_and_set_matching_fn_cb([&]() {
      return mf::build::detail::build_multi_function_with_n_inputs_one_output<float4>(
          "AgX View Transform",
          [=, &builder](const float4 &color,
                       const float general_contrast_in,
                       const float toe_contrast_in,
                       const float shoulder_contrast_in,
                       const float pivot_offset_in,
                       const float log2_min_in,
                       const float log2_max_in,
                       const float3 hue_flights_in,
                       const float3 attenuation_rates_in,
                       const float per_channel_hue_flight_in,
                       const float tinting_scale_in,
                       const float tinting_hue_in,
                       const bool compensate_negatives_in) -> float4 {
            return agx_image_formation(
                color,
                general_contrast_in,
                toe_contrast_in,
                shoulder_contrast_in,
                pivot_offset_in,
                log2_min_in,
                log2_max_in,
                hue_flights_in,
                attenuation_rates_in,
                make_float3(0, 0, 0), /* reverse_hue_flights_in */
                make_float3(0, 0, 0), /* restore_purity_in */
                per_channel_hue_flight_in,
                tinting_scale_in,
                tinting_hue_in,
                compensate_negatives_in,
                builder.node().custom2,
                builder.node().custom3,
                builder.node().custom4,
                builder.node().custom1);
          },
          mf::build::exec_presets::SomeSpanOrSingle<0>(),
          TypeSequence<float4,
                       float,
                       float,
                       float,
                       float,
                       float,
                       float,
                       float3,
                       float3,
                       float,
                       float,
                       float,
                       bool>());
    });
  } else if (!use_inverse_inset && !use_generic_log2) {
    builder.construct_and_set_matching_fn_cb([&]() {
      return mf::build::detail::build_multi_function_with_n_inputs_one_output<float4>(
          "AgX View Transform",
          [=, &builder](const float4 &color,
                       const float general_contrast_in,
                       const float toe_contrast_in,
                       const float shoulder_contrast_in,
                       const float pivot_offset_in,
                       const float3 hue_flights_in,
                       const float3 attenuation_rates_in,
                       const float3 reverse_hue_flights_in,
                       const float3 restore_purity_in,
                       const float per_channel_hue_flight_in,
                       const float tinting_scale_in,
                       const float tinting_hue_in,
                       const bool compensate_negatives_in) -> float4 {
            return agx_image_formation(
                color,
                general_contrast_in,
                toe_contrast_in,
                shoulder_contrast_in,
                pivot_offset_in,
                -10.0f, /* log2_min_in */
                6.5f,   /* log2_max_in */
                hue_flights_in,
                attenuation_rates_in,
                reverse_hue_flights_in,
                restore_purity_in,
                per_channel_hue_flight_in,
                tinting_scale_in,
                tinting_hue_in,
                compensate_negatives_in,
                builder.node().custom2,
                builder.node().custom3,
                builder.node().custom4,
                builder.node().custom1);
          },
          mf::build::exec_presets::SomeSpanOrSingle<0>(),
          TypeSequence<float4,
                       float,
                       float,
                       float,
                       float,
                       float3,
                       float3,
                       float3,
                       float3,
                       float,
                       float,
                       float,
                       bool>());
    });
  } else { /* use_inverse_inset && !use_generic_log2 */
    builder.construct_and_set_matching_fn_cb([&]() {
      return mf::build::detail::build_multi_function_with_n_inputs_one_output<float4>(
          "AgX View Transform",
          [=, &builder](const float4 &color,
                       const float general_contrast_in,
                       const float toe_contrast_in,
                       const float shoulder_contrast_in,
                       const float pivot_offset_in,
                       const float3 hue_flights_in,
                       const float3 attenuation_rates_in,
                       const float per_channel_hue_flight_in,
                       const float tinting_scale_in,
                       const float tinting_hue_in,
                       const bool compensate_negatives_in) -> float4 {
            return agx_image_formation(
                color,
                general_contrast_in,
                toe_contrast_in,
                shoulder_contrast_in,
                pivot_offset_in,
                -10.0f, /* log2_min_in */
                6.5f,   /* log2_max_in */
                hue_flights_in,
                attenuation_rates_in,
                make_float3(0, 0, 0), /* reverse_hue_flights_in */
                make_float3(0, 0, 0), /* restore_purity_in */
                per_channel_hue_flight_in,
                tinting_scale_in,
                tinting_hue_in,
                compensate_negatives_in,
                builder.node().custom2,
                builder.node().custom3,
                builder.node().custom4,
                builder.node().custom1);
          },
          mf::build::exec_presets::SomeSpanOrSingle<0>(),
          TypeSequence<float4,
                       float,
                       float,
                       float,
                       float,
                       float3,
                       float3,
                       float,
                       float,
                       float,
                       bool>());
    });
  }
}

// Registration Function
static void node_register()
{
  static blender::bke::bNodeType ntype;

  cmp_node_type_base(&ntype, "CompositorNodeAgXViewTransform");
  ntype.ui_name = "AgX View Transform";
  ntype.ui_description = "Applies AgX Picture Formation that converts rendered RGB exposure into an Image for Display";
  ntype.enum_name_legacy = "AGX_VIEW_TRANSFORM";
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.declare = node_declare;
  ntype.updatefunc = node_update;
  ntype.initfunc = node_init;
  blender::bke::node_type_size(ntype, 180, 150, 240);
  ntype.build_multi_function = node_build_multi_function;
  ntype.gpu_fn = node_gpu_material;
  blender::bke::node_register_type(ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_composite_agx_view_transform_cc