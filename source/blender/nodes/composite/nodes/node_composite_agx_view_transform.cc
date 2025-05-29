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
  AGX_WORKING_LOG_LINEAR = 0,
  AGX_WORKING_LOG_ACESCCT = 1,
  AGX_WORKING_LOG_ARRI_LOGC3 = 2,
  AGX_WORKING_LOG_ARRI_LOGC4 = 3,
  AGX_WORKING_LOG_GENERIC_LOG2 = 4,
};

static const EnumPropertyItem agx_primaries_items[] = {
    {int(AGXPrimaries::AGX_PRIMARIES_AP0), "ap0", 0, "ACES2065-1 (AP0)", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AP1), "ap1", 0, "ACEScg (AP1)", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_P3D65), "p3d65", 0, "P3-D65", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_REC709), "rec709", 0, "Rec.709", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_REC2020), "rec2020", 0, "Rec.2020", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AWG3), "awg3", 0, "ARRI Alexa Wide Gamut 3", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_AWG4), "awg4", 0, "ARRI Alexa Wide Gamut 4", ""},
    {int(AGXPrimaries::AGX_PRIMARIES_EGAMUT), "egamut", 0, "FilmLight E-Gamut", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem agx_working_log_items[] = {
    {int(AGXWorkingLog::AGX_WORKING_LOG_LINEAR), "linear", 0, "Linear", ""},
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
      "Working Primaries",
      "The working primaries that the AgX mechanism applies to",
      agx_primaries_items, 
      NOD_inline_enum_accessors(custom1),
      int(AGXPrimaries::AGX_PRIMARIES_REC2020)); 

  prop = RNA_def_node_enum(
      srna,
      "working_log",
      "Working Log",
      "The Log curve applied before the sigmoid in the AgX mechanism",
      agx_working_log_items,
      NOD_inline_enum_accessors(custom2),
      int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2)); 

  prop = RNA_def_node_enum(
      srna,
      "display_primaries",
      "Display Primaries",
      "The primaries of the target display device",
      agx_primaries_items,
      NOD_inline_enum_accessors(custom3),
      int(AGXPrimaries::AGX_PRIMARIES_REC709));

  prop = RNA_def_node_boolean(
      srna,
      "sync_outset_to_inset",
      "Use Same Settings for Restoration",
      "Use the same settings as Attenuation section for Purity Restoration, for ease of use",
      NOD_inline_enum_accessors(custom4),
      bool false);
}

// initialize
static void node_init(bNodeTree * /*tree*/, bNode *node) {
  node->custom1 =  int(AGXPrimaries::AGX_PRIMARIES_REC2020);
  node->custom2 = int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2);
  node->custom3 = int(AGXPrimaries::AGX_PRIMARIES_REC709);
  node->custom4 = bool false;
}

// Node Declaration
static void node_declare(NodeDeclarationBuilder &b) {
  b.use_custom_socket_order();
  b.allow_any_socket_order();
  b.add_default_layout();

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
    .subtype(PROP_NONE)
    .description(
        "Slope of the S curve."
        "Slope of the S curve. Controls the general contrast across the image");

  curve_panel.add_input<decl::Float>("Toe Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Toe exponential power of the S curve."
        "Higher values make darker regions crush harder towards black");

  curve_panel.add_input<decl::Float>("Shoulder Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Shoulder exponential power of the S curve."
        "Higher values make brighter regions crush harder towards white");

  curve_panel.add_input<decl::Float>("Contrast Pivot Offset")
    .default_value(0.0f)
    .min(-0.3f)
    .max(0.18f)
    .subtype(PROP_NONE)
    .short_label("Pivot Offset")
    .description(
        "Controls the pivot point for all contrast adjustments");

  curve_panel.add_input<decl::Float>("Log2 Minimum Exposure")
    .default_value(-10.0f)
    .min(-15.0f)
    .max(-5.0f)
    .subtype(PROP_NONE)
    .description(
        "The lower end of the generic log2 curve. Values are in Exposure stops."
        "Only in use when working log is set to Generic Log2");

  curve_panel.add_input<decl::Float>("Log2 Maximum Exposure")
    .default_value(6.5f)
    .min(2.8f)
    .max(38.0f)
    .subtype(PROP_NONE)
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

  b.add_input<decl::Bool>("Compensate for the Negatives")
    .default_value(true)
    .description(
        "Use special luminance compensation technique to prevent out-of-gamut negative values."
        "Done in both pre-curve and post-curve state.");
}

// Put Enums on UI Layout
static void node_layout(uiLayout *layout, bContext * /*C*/, PointerRNA *ptr)
{
  // Draw the "working_primaries" enum property
  layout->prop(ptr,
    "working_primaries",
    UI_ITEM_R_SPLIT_EMPTY_NAME,
    std::nullopt,
    ICON_NONE);
  // Draw the "working_log" enum property
  layout->prop(ptr,
    "working_log",
    UI_ITEM_R_SPLIT_EMPTY_NAME,
    std::nullopt,
    ICON_NONE);

  // Draw the "display_primaries" enum property
  layout->prop(ptr,
    "display_primaries",
    UI_ITEM_R_SPLIT_EMPTY_NAME,
    std::nullopt,
    ICON_NONE);
  
  inset_panel.add_layout([](uiLayout *layout, bContext * /*C*/, PointerRNA *ptr) {
    layout->prop(ptr,
      "sync_outset_to_inset",
      UI_ITEM_R_SPLIT_EMPTY_NAME,
      std::nullopt,
      ICON_NONE);
  }
  );

}

static void node_update(bNodeTree *ntree, bNode *node)
{
  // Get the value of the boolean property
  bool use_same_settings = node->custom4
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
  bool log2_settings_available = node->custom2 == int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2);
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

// Multi-function Builder
class AgXViewTransformFunction : public mf::MultiFunction {
 public:
  AGXPrimaries p_working_primaries;
  AGXWorkingLog p_working_log;
  AGXPrimaries p_display_primaries;
  bool p_use_inverse_inset_in;

  explicit AgXViewTransformFunction(const bNode &node) {
    p_working_primaries = static_cast<AGXPrimaries>(node.custom1);
    p_working_log = static_cast<AGXWorkingLog>(node.custom2);
    p_display_primaries = static_cast<AGXPrimaries>(node.custom3);
    p_use_inverse_inset_in = node.custom4;
    
    static const mf::Signature signature = []() {
      mf::Signature sig;
      mf::SignatureBuilder builder("AgXViewTransform", sig);
      // Socket Inputs:
      builder.single_input<float4>("Color");                            // Index 0
      builder.single_input<float>("General Contrast");                  // Index 1
      builder.single_input<float>("Toe Contrast");                      // Index 2
      builder.single_input<float>("Shoulder Contrast");                 // Index 3
      builder.single_input<float>("Contrast Pivot Offset");             // Index 4
      builder.single_input<float>("Log2 Minimum Exposure");             // Index 5
      builder.single_input<float>("Log2 Maximum Exposure");             // Index 6
      builder.single_input<float3>("Hue Flights");                      // Index 7
      builder.single_input<float3>("Attenuation Rates");                // Index 8
      builder.single_input<float3>("Reverse Hue Flights");              // Index 9
      builder.single_input<float3>("Restore Purity");                   // Index 10
      builder.single_input<float>("Per-Channel Hue Flight");            // Index 11
      builder.single_input<float>("Tinting Scale");                     // Index 12
      builder.single_input<float>("Tinting Hue");                       // Index 13
      builder.single_input<bool>("Compensate for the Negatives");       // Index 14
      // Output:
      builder.single_output<float4>("Color");                           // Index 15
      return sig;
    }();
    this->set_signature(&signature);
  }

  void call(const IndexMask &mask, mf::Params params, mf::Context /*context*/) const override {
    const VArray<float4> in_color = params.readonly_single_input<float4>(0, "Color");
    const VArray<float> general_contrast_in = params.readonly_single_input<float>(1, "General Contrast");
    const VArray<float> toe_contrast_in = params.readonly_single_input<float>(2, "Toe Contrast");
    const VArray<float> shoulder_contrast_in = params.readonly_single_input<float>(3, "Shoulder Contrast");
    const VArray<float> pivot_offset_in = params.readonly_single_input<float>(4, "Contrast Pivot Offset");
    const VArray<float> log2_min_in = params.readonly_single_input<float>(5, "Log2 Minimum Exposure");
    const VArray<float> log2_max_in = params.readonly_single_input<float>(6, "Log2 Maximum Exposure");
    const VArray<float3> hue_flights_in = params.readonly_single_input<float3>(7, "Hue Flights");
    const VArray<float3> attenuation_rates_in = params.readonly_single_input<float3>(8, "Attenuation Rates");
    const VArray<float3> reverse_hue_flights_in = params.readonly_single_input<float3>(9, "Reverse Hue Flights");
    const VArray<float3> restore_purity_in = params.readonly_single_input<float3>(10, "Restore Purity");
    const VArray<float> per_channel_hue_flight_in = params.readonly_single_input<float>(11, "Per-Channel Hue Flight");
    const VArray<float> tinting_scale_in = params.readonly_single_input<float>(12, "Tinting Scale");
    const VArray<float> tinting_hue_in = params.readonly_single_input<float>(13, "Tinting Hue");
    const VArray<bool> compensate_negatives_in = params.readonly_single_input<bool>(14, "Compensate for the Negatives");

    MutableSpan<float4> out_color = params.uninitialized_single_output<float4>(15, "Color");

    mask.foreach_index([&](const int64_t i) {
      float4 col = in_color[i];
      float alpha = col.w;
      float in_rgb_array[3] = {col.x, col.y, col.z};
      float in_xyz_array[3];
      IMB_colormanagement_scene_linear_to_xyz(in_xyz_array, in_rgb_array);
      float3 in_xyz = make_float3(in_xyz_array[0], in_xyz_array[1], in_xyz_array[2]);

      float3x3 xyz_to_working = XYZtoRGB(COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);
      float3 rgb = mult_f3_f33(in_xyz, xyz_to_working);

      // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
      if (compensate_negatives_in[i]) {
        rgb = compensate_low_side(rgb, false, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);
      }
      else {
        rgb = maxf3(0, rgb);
      }

      // generate inset matrix
      Chromaticities inset_chromaticities = InsetPrimaries(
          COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
          attenuation_rates_in[i].x, attenuation_rates_in[i].y, attenuation_rates_in[i].z,
          hue_flights_in[i].x, hue_flights_in[i].y, hue_flights_in[i].z);

      float3x3 insetmat = RGBtoRGB(inset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);

      // apply inset matrix
      rgb = mult_f3_f33(rgb, insetmat);

      // record pre-formation chromaticity angle
      float3 pre_curve_hsv;
      rgb_to_hsv_v(rgb, pre_curve_hsv);

      // encode to working log
      rgb = lin2log(rgb, static_cast<int>(p_working_log), log2_min_in[i], log2_max_in[i]);

      // apply sigmoid, the image is formed at this point
      float log_midgray = lin2log(make_float3(0.18f, 0.18f, 0.18f), static_cast<int>(p_working_log), log2_min_in[i], log2_max_in[i]).x;
      float image_native_power = 2.4f;
      float midgray = pow(0.18f, 1.0f / image_native_power);
      rgb.x = sigmoid(rgb.x, shoulder_contrast_in[i], toe_contrast_in[i], general_contrast_in[i], log_midgray + pivot_offset_in[i], midgray);
      rgb.y = sigmoid(rgb.y, shoulder_contrast_in[i], toe_contrast_in[i], general_contrast_in[i], log_midgray + pivot_offset_in[i], midgray);
      rgb.z = sigmoid(rgb.z, shoulder_contrast_in[i], toe_contrast_in[i], general_contrast_in[i], log_midgray + pivot_offset_in[i], midgray);
      float3 img = rgb;
      // Linearize the formed image assuming its native transfer function is Rec.1886 curve
      img = spowf3(img, image_native_power);

      // lerp pre- and post-curve chromaticity angle
      float3 post_curve_hsv;
      rgb_to_hsv_v(img, post_curve_hsv);
      post_curve_hsv[0] = lerp_chromaticity_angle(pre_curve_hsv[0], post_curve_hsv[0], per_channel_hue_flight_in[i]);
      hsv_to_rgb_v(post_curve_hsv, img);

      // generate outset matrix
      float3x3 outsetmat;
      if (p_use_inverse_inset_in[i]) {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
            attenuation_rates_in[i].x, attenuation_rates_in[i].y, attenuation_rates_in[i].z, // Uses attenuation settings
            hue_flights_in[i].x, hue_flights_in[i].y, hue_flights_in[i].z,         // Uses attenuation settings
            tinting_hue_in[i] + 180, tinting_scale_in[i]);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
      }
      else {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
            restore_purity_in[i].x, restore_purity_in[i].y, restore_purity_in[i].z,
            reverse_hue_flights_in[i].x, reverse_hue_flights_in[i].y, reverse_hue_flights_in[i].z,
            tinting_hue_in[i] + 180, tinting_scale_in[i]);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
      }

      // apply outset matrix
      img = mult_f3_f33(img, outsetmat);

      // convert from working primaries to target display primaries
      float3x3 working_to_display = RGBtoRGB(COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
                                             COLOR_SPACE_PRI[static_cast<int>(p_display_primaries)]);
      img = mult_f3_f33(img, working_to_display);

      // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
      if (compensate_negatives_in[i]) {
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
      col.x = img_array[0];
      col.y = img_array[1];
      col.z = img_array[2];
      col.w = alpha;

      out_color[i] = col;
    });
  }
};

// Multi-function Builder
static void node_build_multi_function(NodeMultiFunctionBuilder &builder) {
  builder.construct_and_set_matching_fn<AgXViewTransformFunction>(builder.node());
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
  blender::bke::node_type_size_preset(ntype, blender::bke::eNodeSizePreset::Large);
  ntype.draw_buttons = node_layout;
  ntype.build_multi_function = node_build_multi_function;
  blender::bke::node_register_type(ntype);

  node_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_composite_agx_view_transform_cc
