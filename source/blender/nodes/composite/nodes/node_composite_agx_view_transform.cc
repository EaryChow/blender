/* SPDX-FileCopyrightText: 2025 Blender Authors */
/* SPDX-License-Identifier: GPL-2.0-or-later */

// Include Headers
#include "BKE_node.hh"
#include "BLI_math_base.hh"
#include "BLI_math_color.h"
#include "BLI_math_color.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "COM_node_operation.hh"
#include "DNA_node_types.h"
#include "FN_multi_function_builder.hh"
#include "IMB_colormanagement.hh"
#include "NOD_node_declaration.hh"
#include "node_composite_util.hh"
#include "node_cmp_agx_utils.hh"
#include "NOD_multi_function.hh"
#include "RNA_access.hh"
#include "UI_interface.hh"
// Namespace Declaration
namespace blender::nodes::node_composite_agx_view_transform_cc {

// Storage Structure
struct NodeAgxViewTransform {
};
NODE_STORAGE_FUNCS(NodeAgxViewTransform);

// define enums
static const EnumPropertyItem agx_primaries_items[] = {
    {AGX_PRIMARIES_AP0, "ap0", 0, "ACES2065-1 (AP0)", ""},
    {AGX_PRIMARIES_AP1, "ap1", 0, "ACEScg (AP1)", ""},
    {AGX_PRIMARIES_P3D65, "p3d65", 0, "P3-D65", ""},
    {AGX_PRIMARIES_REC709, "rec709", 0, "Rec.709", ""},
    {AGX_PRIMARIES_REC2020, "rec2020", 0, "Rec.2020", ""},
    {AGX_PRIMARIES_AWG3, "awg3", 0, "ARRI Alexa Wide Gamut 3", ""},
    {AGX_PRIMARIES_AWG4, "awg4", 0, "ARRI Alexa Wide Gamut 4", ""},
    {AGX_PRIMARIES_EGAMUT, "egamut", 0, "FilmLight E-Gamut", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem agx_working_log_items[] = {
    {AGX_WORKING_LOG_LINEAR, "linear", 0, "Linear", ""},
    {AGX_WORKING_LOG_ACESCCT, "acescct", 0, "ACEScct", ""},
    {AGX_WORKING_LOG_ARRI_LOGC3, "arri_logc3", 0, "ARRI LogC3", ""},
    {AGX_WORKING_LOG_ARRI_LOGC4, "arri_logc4", 0, "ARRI LogC4", ""},
    {AGX_WORKING_LOG_GENERIC_LOG2, "generic_log2", 0, "Generic Log2", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

// Node Declaration
static void cmp_node_agx_view_transform_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>("Color")
     .default_value({1.0f, 1.0f, 1.0f, 1.0f})
     .compositor_domain_priority(0);

  /* Panel for Working Space setting. */
  PanelDeclarationBuilder &working_space_panel = b.add_panel("Working Space").default_closed(true);
  working_space_panel.add_input<decl::Int>("Working Primaries")
    .enum_items(agx_primaries_items)
    .default_value(AGX_PRIMARIES_REC2020)
    .description("The working primaries we apply the AgX mechanism to");

  /* Panel for working log setting. */
  PanelDeclarationBuilder &working_log_panel = b.add_panel("Working Log").default_closed(true);
  working_log_panel.add_input<decl::Int>("Working Log")
    .enum_items(agx_working_log_items)
    .default_value(AGX_WORKING_LOG_GENERIC_LOG2)
    .description("The Log curve applied before the sigmoid in the AgX mechanism");

  /* Panel for log and sigmoid curve settings. */
  PanelDeclarationBuilder &curve_panel = b.add_panel("Curve").default_closed(false);

  curve_panel.add_input<decl::Float>("General Contrast")
    .default_value(2.4f)
    .min(1.4f)
    .max(4.0f)
    .subtype(PROP_NONE)
    .description(
        "Slope for the curve function."
        "Controls the general contrast across the image");
      
  curve_panel.add_input<decl::Float>("Toe Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Exponential power of toe for the curve function."
        "Higher values make dark regions crush harder towards black");
      
  curve_panel.add_input<decl::Float>("Shoulder Contrast")
    .default_value(1.5f)
    .min(0.7f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .description(
        "Exponential power of shoulder for the curve function."
        "Higher values make bright regions crush harder towards white");
      
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
        "Percentage relative to the primary chromaticity purity."
        "by which the chromaticity scales inwards before curve");
     
  /* Panel for outset matrix settings. */
  PanelDeclarationBuilder &outset_panel = b.add_panel("Purity Restoration").default_closed(true);

  outset_panel.add_input<decl::Bool>("Use Same Settings as Attenuation")
    .default_value(false)
    .description(
        "Use the same settings as Attenuation section for ease of use."
        "Turning this on will disable the following two settings");

  outset_panel.add_input<decl::Vector>("Reverse Hue Flights")
    .default_value({0.0f, 0.0f, 0.0f})
    .min(-10.0f)
    .max(10.0f)
    .subtype(PROP_NONE)
    .short_label("Rotation")
    .description(
        "Hue Rotation angle in degrees for each of the RGB primaries after curve."
        "Direction is the reverse of the Attenuation. Negative is counterclockwise, positive is clockwise.");
     
  outset_panel.add_input<decl::Vector>("Restore Purity")
    .default_value({0.323174f, 0.283256f, 0.037433f})
    .min(0.0f)
    .max(0.6f)
    .subtype(PROP_NONE)
    .description(
        "Percentage relative to the primary chromaticity purity."
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
  
   /* Panel for Output Gamut Limit settings. */
  PanelDeclarationBuilder &display_gamut_panel = b.add_panel("Display Primaries").default_closed(true);

  display_gamut_panel.add_input<decl::Int>("Display Primaries")
    .enum_items(agx_primaries_items)
    .default_value(AGX_PRIMARIES_REC709)
    .description("The primaries of the target display device");

  b.add_input<decl::Bool>("Compensate for the Negatives")
    .default_value(true)
    .description(
        "Use special luminance compensation technique to prevent out-of-gamut negative values."
        "Done in both pre-curve and post-curve state.");
  
  b.add_output<decl::Color>("Color");

}

// Multi-function Builder
class AgXViewTransformFunction : public mf::MultiFunction {
 public:
  AgXViewTransformFunction()
  {
    static const mf::Signature signature = []() {
      mf::SignatureBuilder builder{"AgX View Transform"};
      builder.single_input<float4>("Color");
      builder.single_input<int>("Working Primaries");
      builder.single_input<int>("Working Log");
      builder.single_input<float>("General Contrast");
      builder.single_input<float>("Toe Contrast");
      builder.single_input<float>("Shoulder Contrast");
      builder.single_input<float>("Contrast Pivot Offset");
      builder.single_input<float>("Log2 Minimum Exposure");
      builder.single_input<float>("Log2 Maximum Exposure");
      builder.single_input<float3>("Hue Flights");
      builder.single_input<float3>("Attenuation Rates");
      builder.single_input<bool>("Use Same Settings as Attenuation");
      builder.single_input<float3>("Reverse Hue Flights");
      builder.single_input<float3>("Restore Purity");
      builder.single_input<float>("Per-Channel Hue Flight");
      builder.single_input<float>("Tinting Scale");
      builder.single_input<float>("Tinting Hue");
      builder.single_input<int>("Display Primaries");
      builder.single_input<bool>("Compensate for the Negatives");
      builder.single_output<float4>("Color");
      return builder.build();
    }();
    this->set_signature(&signature);
  }

  void call(const IndexMask &mask, mf::Params params, mf::Context /*context*/) const override
  {
    const VArray<float4> in_color = params.readonly_single_input<blender::Color4f>(0, "Color");
    const VArray<int> working_primaries = params.readonly_single_input<int>(1, "Working Primaries");
    const VArray<int> working_log = params.readonly_single_input<int>(2, "Working Log");
    const VArray<float> general_contrast = params.readonly_single_input<float>(3, "General Contrast");
    const VArray<float> toe_contrast = params.readonly_single_input<float>(4, "Toe Contrast");
    const VArray<float> shoulder_contrast = params.readonly_single_input<float>(5, "Shoulder Contrast");
    const VArray<float> pivot_offset = params.readonly_single_input<float>(6, "Contrast Pivot Offset");
    const VArray<float> log2_min = params.readonly_single_input<float>(7, "Log2 Minimum Exposure");
    const VArray<float> log2_max = params.readonly_single_input<float>(8, "Log2 Maximum Exposure");
    const VArray<float3> inset_rotate = params.readonly_single_input<float3>(9, "Hue Flights");
    const VArray<float3> inset_scale = params.readonly_single_input<float3>(10, "Attenuation Rates");
    const VArray<bool> use_inverse_inset = params.readonly_single_input<bool>(11, "Use Same Settings as Attenuation");
    const VArray<float3> outset_rotate = params.readonly_single_input<float3>(12, "Reverse Hue Flights");
    const VArray<float3> outset_scale = params.readonly_single_input<float3>(13, "Restore Purity");
    const VArray<float> per_channel_hue_flight = params.readonly_single_input<float>(14, "Per-Channel Hue Flight");
    const VArray<float> tinting_outset = params.readonly_single_input<float>(15, "Tinting Scale");
    const VArray<float> tinting_rotate = params.readonly_single_input<float>(16, "Tinting Hue");
    const VArray<int> display_primaries = params.readonly_single_input<int>(17, "Display Primaries");
    const VArray<bool> compensate_negatives = params.readonly_single_input<bool>(18, "Compensate for the Negatives");

    MutableSpan<float4> out_color = params.uninitialized_single_output<blender::Color4f>(19, "Color");

    mask.foreach_index([&](const int64_t i) {
      Color4f col = in_color[i];
      // save alpha channel for direct output, we only process RGB here
      float alpha = col.a;
      // use color management system to import colors from OCIO's scene_linear role space
      float in_rgb_array[3] = {col.r, col.g, col.b};
      float in_xyz_array[3];
      IMB_colormanagement_scene_linear_to_xyz(in_xyz_array, in_rgb_array);
      float3 in_xyz = make_float3(in_xyz_array[0], in_xyz_array[1], in_xyz_array[2]);
      float3x3 xyz_to_working = XYZtoRGB(COLOR_SPACE_PRI[working_primaries[i]]);
      float3 rgb = mult_f3_f33(in_xyz, xyz_to_working);

      // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
      if (compensate_negatives[i]) {
        rgb = compensate_low_side(rgb, false, COLOR_SPACE_PRI[working_primaries[i]]);
      }
      else {
        rgb = maxf3(0, rgb);
      }
      // generate inset matrix
      Chromaticities inset_chromaticities = InsetPrimaries(
          COLOR_SPACE_PRI[working_primaries[i]],
          inset_scale[i].x, inset_scale[i].y, inset_scale[i].z,
          inset_rotate[i].x, inset_rotate[i].y, inset_rotate[i].z);

      float3x3 insetmat = RGBtoRGB(inset_chromaticities, COLOR_SPACE_PRI[working_primaries[i]]);
      // apply inset matrix
      rgb = mult_f3_f33(rgb, insetmat);

      // record pre-formation chromaticity angle
      float3 pre_curve_hsv;
      pre_curve_hsv = rgb_to_hsv(rgb);

      // encode to working log
      rgb = lin2log(rgb, working_log[i], log2_min[i], log2_max[i]);

      // apply sigmoid, the image is formed at this point
      float log_midgray = lin2log(make_float3(0.18f), working_log[i], log2_min[i], log2_max[i]).x;
      float image_native_power = 2.4f; // assume image's native transfer function is Rec.1886 power 2.4
      float midgray = pow(0.18f, 1.0f / image_native_power);
      rgb.x = sigmoid(rgb.x, shoulder_contrast[i], toe_contrast[i], general_contrast[i], log_midgray + pivot_offset[i], midgray);
      rgb.y = sigmoid(rgb.y, shoulder_contrast[i], toe_contrast[i], general_contrast[i], log_midgray + pivot_offset[i], midgray);
      rgb.z = sigmoid(rgb.z, shoulder_contrast[i], toe_contrast[i], general_contrast[i], log_midgray + pivot_offset[i], midgray);
      float3 img = rgb;
      // Linearize the formed image assuming it's native transfer function is Rec.1886 curve
      img = spowf3(img, image_native_power);

      // lerp pre- and post-curve chromaticity angle
      float3 post_curve_hsv;
      rgb_to_hsv_v(img, post_curve_hsv);
      post_curve_hsv[0] = lerp_chromaticity_angle(pre_curve_hsv[0], post_curve_hsv[0], per_channel_hue_flight[i]);
      img = hsv_to_rgb(post_curve_hsv);

      // generate outset matrix
      float3x3 outsetmat;
      if (use_inverse_inset[i]) {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[working_primaries[i]],
            inset_scale[i].x, inset_scale[i].y, inset_scale[i].z,
            inset_rotate[i].x, inset_rotate[i].y, inset_rotate[i].z,
            tinting_rotate[i] + 180, tinting_outset[i]);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[working_primaries[i]]));
      }
      else {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[working_primaries[i]],
            outset_scale[i].x, outset_scale[i].y, outset_scale[i].z,
            outset_rotate[i].x, outset_rotate[i].y, outset_rotate[i].z,
            tinting_rotate[i] + 180, tinting_outset[i]);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[working_primaries[i]]));
      }

      // apply outset matrix
      img = mult_f3_f33(img, outsetmat);

      // convert from working primaries to target display primaries
      float3x3 working_to_display = RGBtoRGB(COLOR_SPACE_PRI[working_primaries[i]], COLOR_SPACE_PRI[display_primaries[i]]);
      img = mult_f3_f33(img, working_to_display);

      // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
      if (compensate_negatives[i]) {
        img = compensate_low_side(img, true, COLOR_SPACE_PRI[display_primaries[i]]);
      }
      else {
        img = maxf3(0, img);
      }

      // convert linearized formed image back to OCIO's scene_linear role space
      float3x3 display_to_xyz = RGBtoXYZ(COLOR_SPACE_PRI[display_primaries[i]]);
      float3 out_xyz = mult_f3_f33(img, display_to_xyz);
      float out_xyz_array[3] = {out_xyz.x, out_xyz.y, out_xyz.z};
      float img_array[3];
      IMB_colormanagement_xyz_to_scene_linear(img_array, out_xyz_array);

      // re-combine the alpha channel
      col.r = img_array[0];
      col.g = img_array[1];
      col.b = img_array[2];
      col.a = alpha;

      out_color[i] = col;
    });
  }
};

// Multi-function Builder
static void cmp_node_agx_view_transform_build_multi_function(NodeMultiFunctionBuilder &builder)
{
  static auto fn = blender::mf::build::SI1_SO<AgXViewTransformFunction, blender::Color4f, blender::Color4f>("AgX View Transform");
  builder.set_matching_fn(fn);
}

}  // namespace blender::nodes::node_composite_agx_view_transform_cc

// Registration Function
static void register_node_type_cmp_node_agx_view_transform()
{
  namespace file_ns = blender::nodes::node_composite_agx_view_transform_cc;
  static blender::bke::bNodeType ntype;

  cmp_node_type_base(&ntype, "CompositorNodeAgXViewTransform");
  ntype.ui_name = "AgX View Transform";
  ntype.ui_description = "Applies AgX Picture Formation that converts rendered RGB exposure into an Image for Display";
  ntype.idname = "CompositorNodeAgXViewTransform";
  ntype.enum_name_legacy = nullptr;
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.declare = file_ns::cmp_node_agx_view_transform_declare;
  ntype.build_multi_function = file_ns::cmp_node_agx_view_transform_build_multi_function;

  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(register_node_type_cmp_node_agx_view_transform)