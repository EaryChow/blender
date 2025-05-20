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
enum AGXPrimaries {
  AGX_PRIMARIES_AP0 = 0,
  AGX_PRIMARIES_AP1,
  AGX_PRIMARIES_P3D65,
  AGX_PRIMARIES_REC709,
  AGX_PRIMARIES_REC2020,
  AGX_PRIMARIES_AWG3,
  AGX_PRIMARIES_AWG4,
  AGX_PRIMARIES_EGAMUT,
};

enum AGXWorkingLog {
  AGX_WORKING_LOG_LINEAR = 0,
  AGX_WORKING_LOG_ACESCCT,
  AGX_WORKING_LOG_ARRI_LOGC3,
  AGX_WORKING_LOG_ARRI_LOGC4,
  AGX_WORKING_LOG_GENERIC_LOG2,
};

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

// Storage Structure
struct NodeAgxViewTransform {
  AGXPrimaries working_primaries;
  AGXWorkingLog working_log;
  AGXPrimaries display_primaries;
};
NODE_STORAGE_FUNCS(NodeAgxViewTransform);

// Storage free/copy functions
static void node_free_agx_storage(bNode *node) {
    MEM_SAFE_FREE(node->storage);
}

static void node_copy_agx_storage(
    bNodeTree * /*dest_ntree*/,
    bNode *dest_node,
    const bNode *src_node)
{
  if (src_node->storage) {
    dest_node->storage = MEM_dupallocN(src_node->storage);
  }
  else {
    dest_node->storage = nullptr;
  }
}

// --- Custom Accessor Functions for Enum Properties ---
static int rna_AgxNode_working_primaries_get(PointerRNA *ptr, PropertyRNA * /*prop*/) {
  const bNode &node = *static_cast<const bNode *>(ptr->data);
  return static_cast<int>(node_storage(node).working_primaries);
}
static void rna_AgxNode_working_primaries_set(PointerRNA *ptr, PropertyRNA * /*prop*/, const int value) {
  bNode &node = *static_cast<bNode *>(ptr->data);
  node_storage(node).working_primaries = static_cast<AGXPrimaries>(value); // Explicit cast
}

static int rna_AgxNode_working_log_get(PointerRNA *ptr, PropertyRNA * /*prop*/) {
  const bNode &node = *static_cast<const bNode *>(ptr->data);
  return static_cast<int>(node_storage(node).working_log);
}
static void rna_AgxNode_working_log_set(PointerRNA *ptr, PropertyRNA * /*prop*/, const int value) {
  bNode &node = *static_cast<bNode *>(ptr->data);
  node_storage(node).working_log = static_cast<AGXWorkingLog>(value);
}

static int rna_AgxNode_display_primaries_get(PointerRNA *ptr, PropertyRNA * /*prop*/) {
  const bNode &node = *static_cast<const bNode *>(ptr->data);
  return static_cast<int>(node_storage(node).display_primaries);
}
static void rna_AgxNode_display_primaries_set(PointerRNA *ptr, PropertyRNA * /*prop*/, const int value) {
  bNode &node = *static_cast<bNode *>(ptr->data);
  node_storage(node).display_primaries = static_cast<AGXPrimaries>(value);
}
// --- End of Custom Accessor Functions ---

// RNA functions for node properties
static void cmp_node_agx_view_transform_rna(StructRNA *srna) {
  PropertyRNA *prop;

  // For working_primaries using custom accessors
  EnumRNAAccessors working_primaries_accessors(
      rna_AgxNode_working_primaries_get,
      rna_AgxNode_working_primaries_set
  );
  prop = RNA_def_node_enum(
      srna,
      "working_primaries",
      "Working Primaries",
      "The working primaries that the AgX mechanism applies to",
      agx_primaries_items,
      working_primaries_accessors,
      AGX_PRIMARIES_REC2020
  );

  // For working_log using custom accessors
  EnumRNAAccessors working_log_accessors(
      rna_AgxNode_working_log_get,
      rna_AgxNode_working_log_set
  );
  prop = RNA_def_node_enum(
      srna,
      "working_log",
      "Working Log",
      "The Log curve applied before the sigmoid in the AgX mechanism",
      agx_working_log_items,
      working_log_accessors,
      AGX_WORKING_LOG_GENERIC_LOG2
  );

  // For display_primaries using custom accessors
  EnumRNAAccessors display_primaries_accessors(
      rna_AgxNode_display_primaries_get,
      rna_AgxNode_display_primaries_set
  );
  prop = RNA_def_node_enum(
      srna,
      "display_primaries",
      "Display Primaries",
      "The primaries of the target display device",
      agx_primaries_items,
      display_primaries_accessors,
      AGX_PRIMARIES_REC709
  );
}

// initialize
static void cmp_node_agx_view_transform_init(bNodeTree * /*tree*/, bNode *node) {
  NodeAgxViewTransform *data = MEM_callocN<NodeAgxViewTransform>(__func__);
  data->working_primaries = AGX_PRIMARIES_REC2020;
  data->working_log = AGX_WORKING_LOG_GENERIC_LOG2;
  data->display_primaries = AGX_PRIMARIES_REC709;
  node->storage = data;
}

// Node Declaration
static void cmp_node_agx_view_transform_declare(NodeDeclarationBuilder &b) {
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

  outset_panel.add_input<decl::Bool>("Use Same Settings as Attenuation")
    .default_value(false)
    .description("Use the same settings as Attenuation section for Purity Restoration, for ease of use");

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

  b.add_output<decl::Color>("Color");
}

// Put Enums on UI Layout
static void cmp_node_agx_view_transform_layout(uiLayout *layout,
                                               bContext * /*C*/,
                                               PointerRNA *ptr)
{
// Draw the "working_primaries" enum property
layout->prop(ptr,
             "working_primaries",
             UI_ITEM_NONE,   // <<< Use the defined enum member for no/default flags
             std::nullopt,   // Use RNA's ui_name for the label text
             ICON_NONE);     // No icon

// Draw the "working_log" enum property
layout->prop(ptr,
             "working_log",
             UI_ITEM_NONE,
             std::nullopt,
             ICON_NONE);

// Draw the "display_primaries" enum property
layout->prop(ptr,
             "display_primaries",
             UI_ITEM_NONE,
             std::nullopt,
             ICON_NONE);
}


// Multi-function Builder
class AgXViewTransformFunction : public mf::MultiFunction {
 public:
  // Members for "baked-in" settings (enums from node storage)
  AGXPrimaries p_working_primaries;
  AGXWorkingLog p_working_log;
  AGXPrimaries p_display_primaries;

  explicit AgXViewTransformFunction(const bNode &node) {
    const NodeAgxViewTransform *s = static_cast<const NodeAgxViewTransform *>(node.storage);
    p_working_primaries = s->working_primaries;
    p_working_log = s->working_log;
    p_display_primaries = s->display_primaries;

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
      builder.single_input<bool>("Use Same Settings as Attenuation");   // Index 9
      builder.single_input<float3>("Reverse Hue Flights");              // Index 10
      builder.single_input<float3>("Restore Purity");                   // Index 11
      builder.single_input<float>("Per-Channel Hue Flight");            // Index 12
      builder.single_input<float>("Tinting Scale");                     // Index 13
      builder.single_input<float>("Tinting Hue");                       // Index 14
      builder.single_input<bool>("Compensate for the Negatives");       // Index 15
      // Output:
      builder.single_output<float4>("Color");                           // Index 16
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
    const VArray<bool> use_inverse_inset_in = params.readonly_single_input<bool>(9, "Use Same Settings as Attenuation");
    const VArray<float3> reverse_hue_flights_in = params.readonly_single_input<float3>(10, "Reverse Hue Flights");
    const VArray<float3> restore_purity_in = params.readonly_single_input<float3>(11, "Restore Purity");
    const VArray<float> per_channel_hue_flight_in = params.readonly_single_input<float>(12, "Per-Channel Hue Flight");
    const VArray<float> tinting_scale_in = params.readonly_single_input<float>(13, "Tinting Scale");
    const VArray<float> tinting_hue_in = params.readonly_single_input<float>(14, "Tinting Hue");
    const VArray<bool> compensate_negatives_in = params.readonly_single_input<bool>(15, "Compensate for the Negatives");

    MutableSpan<float4> out_color = params.uninitialized_single_output<float4>(16, "Color");

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
      if (use_inverse_inset_in[i]) {
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
static void cmp_node_agx_view_transform_build_multi_function(NodeMultiFunctionBuilder &builder) {
  builder.construct_and_set_matching_fn<AgXViewTransformFunction>(builder.node());
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
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.declare = file_ns::cmp_node_agx_view_transform_declare;
  ntype.updatefunc = nullptr;
  ntype.initfunc = file_ns::cmp_node_agx_view_transform_init;
  ntype.draw_buttons = file_ns::cmp_node_agx_view_transform_layout;
  ntype.build_multi_function = file_ns::cmp_node_agx_view_transform_build_multi_function;
  blender::bke::node_type_storage(
      ntype, "NodeAgxViewTransform", file_ns::node_free_agx_storage, file_ns::node_copy_agx_storage);
  file_ns::cmp_node_agx_view_transform_rna(ntype.rna_ext.srna);
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(register_node_type_cmp_node_agx_view_transform)