/* SPDX-FileCopyrightText: 2025 Blender Authors */
/* SPDX-License-Identifier: GPL-2.0-or-later */

// Include Headers
#include "BLI_math_base.hh"
#include "BLI_math_color.h"
#include "BLI_math_color.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"
#include "BLT_translation.hh"
#include "BKE_node.hh"
#include "COM_node_operation.hh"
#include "DNA_node_types.h"
#include "FN_multi_function_builder.hh"
#include "IMB_colormanagement.hh"
#include "NOD_node_declaration.hh"
#include "node_composite_util.hh"
#include "node_cmp_agx_utils.hh"
#include "NOD_multi_function.hh"
#include "RNA_access.hh"
#include "RNA_define.hh"
#include "UI_interface.hh"
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

// RNA functions node properties
static void cmp_node_agx_view_transform_rna(StructRNA *srna)
{
  PropertyRNA *prop;
  // --- Enum Properties ---
  prop = RNA_def_node_enum(
      srna,
      "working_primaries",
      "Working Primaries",
      "The working primaries that the AgX mechanism applies to",
      agx_primaries_items,
      NOD_storage_enum_accessors(working_primaries),
      AGX_PRIMARIES_REC2020);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_enum(
      srna,
      "working_log",
      "Working Log",
      "The Log curve applied before the sigmoid in the AgX mechanism",
      agx_working_log_items,
      NOD_storage_enum_accessors(working_log),
      AGX_WORKING_LOG_GENERIC_LOG2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_enum(
      srna,
      "display_primaries",
      "Display Primaries",
      "The primaries of the target display device",
      agx_primaries_items,
      NOD_storage_enum_accessors(display_primaries),
      AGX_PRIMARIES_REC709);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  // --- Float Properties ---
  prop = RNA_def_node_float(
      srna,
      "general_contrast",
      "General Contrast",
      "Slope of the S curve. Controls the general contrast across the image",
      NOD_storage_float_accessors(general_contrast), 2.4f, 1.4f, 4.0f);
  RNA_def_property_ui_range(prop, 1.4f, 4.0f, 0.1f, 2); // min, max, step, precision
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "toe_contrast",
      "Toe Contrast",
      "Toe exponential power of the S curve. Higher values make darker regions crush harder towards black",
      NOD_storage_float_accessors(toe_contrast), 1.5f, 0.7f, 10.0f);
  RNA_def_property_ui_range(prop, 0.7f, 10.0f, 0.1f, 2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "shoulder_contrast",
      "Shoulder Contrast",
      "Shoulder exponential power of the S curve. Higher values make brighter regions crush harder towards white",
      NOD_storage_float_accessors(shoulder_contrast), 1.5f, 0.7f, 10.0f);
  RNA_def_property_ui_range(prop, 0.7f, 10.0f, 0.1f, 2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "pivot_offset",
      "Pivot Offset",
      "Controls the pivot point for all contrast adjustments",
      NOD_storage_float_accessors(pivot_offset), 0.0f, -0.3f, 0.18f);
  RNA_def_property_ui_range(prop, -0.3f, 0.18f, 0.01f, 3);
  RNA_def_property_flag(prop, PROP_SHORT_NAME); // For "Short Label" behavior if property name is long
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "log2_min",
      "Log2 Minimum Exposure",
      "The lower end of the generic log2 curve. Values are in Exposure stops. Only in use when working log is set to Generic Log2",
      NOD_storage_float_accessors(log2_min), -10.0f, -15.0f, -5.0f);
  RNA_def_property_ui_range(prop, -15.0f, -5.0f, 0.1f, 2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "log2_max",
      "Log2 Maximum Exposure",
      "The upper end of the generic log2 curve. Values are in Exposure stops. Only in use when working log is set to Generic Log2",
      NOD_storage_float_accessors(log2_max), 6.5f, 2.8f, 38.0f);
  RNA_def_property_ui_range(prop, 2.8f, 38.0f, 0.1f, 2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "per_channel_hue_flight",
      "Per-Channel Hue Flight",
      "The percentage of hue shift introduced by the per-channel curve. Higher value will have yellower orange, for example",
      NOD_storage_float_accessors(per_channel_hue_flight), 0.4f, 0.0f, 1.0f);
  RNA_def_property_subtype(prop, PROP_FACTOR);
  RNA_def_property_ui_range(prop, 0.0f, 1.0f, 0.01f, 2);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "tinting_outset",
      "Tinting Scale",
      "Controls how far the white point shifts in the outset. Affecting the intensity or strength of the tint applied after curve",
      NOD_storage_float_accessors(tinting_outset), 0.0f, -0.2f, 0.2f);
  RNA_def_property_subtype(prop, PROP_FACTOR);
  RNA_def_property_ui_range(prop, -0.2f, 0.2f, 0.01f, 3);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_float(
      srna,
      "tinting_rotate",
      "Tinting Hue",
      "Adjusts the direction in which the white point shifts in the outset. Influencing the overall hue of the tint applied after curve",
      NOD_storage_float_accessors(tinting_rotate), 0.0f, -180.0f, 180.0f);
  RNA_def_property_subtype(prop, PROP_FACTOR);
  RNA_def_property_ui_range(prop, -180.0f, 180.0f, 1.0f, 1);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  // --- Vector Properties ---
  prop = RNA_def_node_vector(
      srna,
      "hue_flights", 3, nullptr,
      "Hue Flights",
      "Hue Rotation angle in degrees for each of the RGB primaries before curve. Negative is clockwise, and positive is counterclockwise",
      NOD_storage_vector_accessors(hue_flights), nullptr); // Last nullptr if default is handled by init
  RNA_def_property_array(prop, 3);
  RNA_def_property_ui_range(prop, -10.0f, 10.0f, 0.1f, 3); // Range for each component
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_vector(
      srna,
      "attenuation_rates", 3, nullptr,
      "Attenuation Rates",
      "Percentage relative to the primary chromaticity purity, by which the chromaticity scales inwards before curve",
      NOD_storage_vector_accessors(attenuation_rates), nullptr);
  RNA_def_property_array(prop, 3);
  RNA_def_property_ui_range(prop, 0.0f, 0.6f, 0.01f, 3);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_vector(
      srna,
      "reverse_hue_flights", 3, nullptr,
      "Reverse Hue Flights",
      "Hue Rotation angle in degrees for each of the RGB primaries after curve. Direction is the reverse of the Attenuation.",
      NOD_storage_vector_accessors(reverse_hue_flights), nullptr);
  RNA_def_property_array(prop, 3);
  RNA_def_property_ui_range(prop, -10.0f, 10.0f, 0.1f, 3);
  RNA_def_property_flag(prop, PROP_SHORT_NAME);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_vector(
      srna,
      "restore_purity", 3, nullptr,
      "Restore Purity",
      "Percentage relative to the primary chromaticity purity, by which the chromaticity scales outwards after curve",
      NOD_storage_vector_accessors(restore_purity), nullptr);
  RNA_def_property_array(prop, 3);
  RNA_def_property_ui_range(prop, 0.0f, 0.6f, 0.01f, 3);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  // --- Boolean Properties ---
  prop = RNA_def_node_boolean(
      srna,
      "use_inverse_inset",
      "Use Same Settings as Attenuation",
      "Use the same settings as Attenuation section for Purity Restoration, for ease of use",
      NOD_storage_bool_accessors(use_inverse_inset), false);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");

  prop = RNA_def_node_boolean(
      srna,
      "compensate_negatives",
      "Compensate for the Negatives",
      "Use special luminance compensation technique to prevent out-of-gamut negative values. Done in both pre-curve and post-curve state",
      NOD_storage_bool_accessors(compensate_negatives), true);
  RNA_def_property_update(prop, NC_NODE | ND_TREE, "rna_Node_update_custom");
}

// Storage Structure
struct NodeAgxViewTransform {
  // Enum properties
  AGXPrimaries working_primaries;
  AGXWorkingLog working_log;
  AGXPrimaries display_primaries;

    // Float properties
  float general_contrast;
  float toe_contrast;
  float shoulder_contrast;
  float pivot_offset; // "Contrast Pivot Offset"
  float log2_min;     // "Log2 Minimum Exposure"
  float log2_max;     // "Log2 Maximum Exposure"
  float per_channel_hue_flight;
  float tinting_outset; // "Tinting Scale"
  float tinting_rotate; // "Tinting Hue"

  // Vector properties
  float hue_flights[3];
  float attenuation_rates[3];
  float reverse_hue_flights[3];
  float restore_purity[3];

  // Boolean properties
  bool use_inverse_inset; // "Use Same Settings as Attenuation"
  bool compensate_negatives; // "Compensate for the Negatives"

};
NODE_STORAGE_FUNCS(NodeAgxViewTransform);

static void node_free_agx_storage(bNode *node) {
  MEM_SAFE_FREE(node->storage);
}

static void node_copy_agx_storage(bNode *nnode, const bNode *node) {
  nnode->storage = MEM_dupallocN(node->storage);
}

// initialize
static void cmp_node_agx_view_transform_init(bNodeTree * /*tree*/, bNode *node)
{
  NodeAgxViewTransform *data = MEM_callocN<NodeAgxViewTransform>(__func__);
  // Enum defaults
  data->working_primaries = AGX_PRIMARIES_REC2020;
  data->working_log = AGX_WORKING_LOG_GENERIC_LOG2;
  data->display_primaries = AGX_PRIMARIES_REC709;

  // Float defaults
  data->general_contrast = 2.4f;
  data->toe_contrast = 1.5f;
  data->shoulder_contrast = 1.5f;
  data->pivot_offset = 0.0f;
  data->log2_min = -10.0f;
  data->log2_max = 6.5f;
  data->per_channel_hue_flight = 0.4f;
  data->tinting_outset = 0.0f;
  data->tinting_rotate = 0.0f;

  // Vector defaults
  float3_to_array(data->hue_flights, {2.13976f, -1.22827f, -3.05174f});
  float3_to_array(data->attenuation_rates, {0.329652f, 0.280513f, 0.124754f});
  float3_to_array(data->reverse_hue_flights, {0.0f, 0.0f, 0.0f});
  float3_to_array(data->restore_purity, {0.323174f, 0.283256f, 0.037433f});

  // Boolean defaults
  data->use_inverse_inset = false;
  data->compensate_negatives = true;

  node->storage = data;
}

// Node Declaration
static void cmp_node_agx_view_transform_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>("Color")
     .default_value({1.0f, 1.0f, 1.0f, 1.0f})
     .compositor_domain_priority(0);

  b.add_output<decl::Color>("Color");

}

// Node UI Layout
static void cmp_node_agx_view_transform_layout(uiLayout *layout, bContext *C, PointerRNA *ptr)
{
  uiLayout *col;

  // Working Space Section
  layout->label(text_ctxt("Working Space"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "working_primaries");
  col->prop(ptr, "working_log", text_ctxt("Log"));

  layout->separator();

  // Curve Section
  layout->label(text_ctxt("Curve"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "general_contrast");
  col->prop(ptr, "toe_contrast");
  col->prop(ptr, "shoulder_contrast");
  col->prop(ptr, "pivot_offset", text_ctxt("Pivot Offset"));
  col->prop(ptr, "log2_min");
  col->prop(ptr, "log2_max");

  layout->separator();

  // Attenuation Section
  layout->label(text_ctxt("Attenuation"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "hue_flights");
  col->prop(ptr, "attenuation_rates");

  layout->separator();

  // Purity Restoration Section
  layout->label(text_ctxt("Purity Restoration"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "use_inverse_inset");

  // Conditionally enable/disable based on use_inverse_inset
  bool use_same = RNA_boolean_get(ptr, "use_inverse_inset");
  uiLayout *sub_col = col->column();
  sub_col->active = !use_same; // Grey out if use_same is true
  sub_col->prop(ptr, "reverse_hue_flights", text_ctxt("Rotation"));
  sub_col->prop(ptr, "restore_purity");

  layout->separator();

  // Look Section
  layout->label(text_ctxt("Look"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "per_channel_hue_flight");
  col->prop(ptr, "tinting_outset", text_ctxt("Tinting Scale"));
  col->prop(ptr, "tinting_rotate", text_ctxt("Tinting Hue"));

  layout->separator();

  // Display Primaries Section
  layout->label(text_ctxt("Target Display Primaries"));
  col = layout->column(UI_LAYOUT_COLUMN_FLOW_FIRST);
  col->prop(ptr, "display_primaries");

  layout->separator();

  // Other settings
  layout->prop(ptr, "compensate_negatives");
}

// Multi-function Builder
class AgXViewTransformFunction : public mf::MultiFunction {
 public:
  // Store ALL global settings as member variables
  AGXPrimaries p_working_primaries;
  AGXWorkingLog p_working_log;
  float p_general_contrast;
  float p_toe_contrast;
  float p_shoulder_contrast;
  float p_pivot_offset;
  float p_log2_min;
  float p_log2_max;
  float3 p_hue_flights;
  float3 p_attenuation_rates;
  bool p_use_inverse_inset;
  float3 p_reverse_hue_flights;
  float3 p_restore_purity;
  float p_per_channel_hue_flight;
  float p_tinting_outset;
  float p_tinting_rotate;
  AGXPrimaries p_display_primaries;
  bool p_compensate_negatives;


  // Constructor that takes the bNode to extract settings from storage
  explicit AgXViewTransformFunction(const bNode &node)
  {
    const NodeAgxViewTransform *s = static_cast<const NodeAgxViewTransform *>(node.storage);

    p_working_primaries = s->working_primaries;
    p_working_log = s->working_log;
    p_display_primaries = s->display_primaries;
    p_general_contrast = s->general_contrast;
    p_toe_contrast = s->toe_contrast;
    p_shoulder_contrast = s->shoulder_contrast;
    p_pivot_offset = s->pivot_offset;
    p_log2_min = s->log2_min;
    p_log2_max = s->log2_max;

    // Convert float arrays from storage to float3 for internal use
    p_hue_flights = make_float3(s->hue_flights);
    p_attenuation_rates = make_float3(s->attenuation_rates);
    p_reverse_hue_flights = make_float3(s->reverse_hue_flights);
    p_restore_purity = make_float3(s->restore_purity);

    p_use_inverse_inset = s->use_inverse_inset;
    p_per_channel_hue_flight = s->per_channel_hue_flight;
    p_tinting_outset = s->tinting_outset;
    p_tinting_rotate = s->tinting_rotate;
    p_compensate_negatives = s->compensate_negatives;

    static const mf::Signature signature = []() {
      mf::Signature sig;
      mf::SignatureBuilder builder("AgXViewTransform", sig);
      builder.single_input<float4>("Color");
      builder.single_output<float4>("Color");
      return sig;
    }();
    this->set_signature(&signature);
  }

  void call(const IndexMask &mask, mf::Params params, mf::Context /*context*/) const override
  {
    const VArray<float4> in_color = params.readonly_single_input<float4>(0, "Color");
    MutableSpan<float4> out_color = params.uninitialized_single_output<float4>(1, "Color");

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
      if (p_compensate_negatives) {
        rgb = compensate_low_side(rgb, false, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);
      }
      else {
        rgb = maxf3(0, rgb);
      }

      // generate inset matrix
      Chromaticities inset_chromaticities = InsetPrimaries(
          COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
          p_attenuation_rates.x, p_attenuation_rates.y, p_attenuation_rates.z,
          p_hue_flights.x, p_hue_flights.y, p_hue_flights.z);

      float3x3 insetmat = RGBtoRGB(inset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]);

      // apply inset matrix
      rgb = mult_f3_f33(rgb, insetmat);

      // record pre-formation chromaticity angle
      float3 pre_curve_hsv;
      rgb_to_hsv_v(rgb, pre_curve_hsv);

      // encode to working log
      rgb = lin2log(rgb, static_cast<int>(p_working_log), p_log2_min, p_log2_max);

      // apply sigmoid, the image is formed at this point
      float log_midgray = lin2log(make_float3(0.18f), static_cast<int>(p_working_log), p_log2_min, p_log2_max).x;
      float image_native_power = 2.4f;
      float midgray = pow(0.18f, 1.0f / image_native_power);
      rgb.x = sigmoid(rgb.x, p_shoulder_contrast, p_toe_contrast, p_general_contrast, log_midgray + p_pivot_offset, midgray);
      rgb.y = sigmoid(rgb.y, p_shoulder_contrast, p_toe_contrast, p_general_contrast, log_midgray + p_pivot_offset, midgray);
      rgb.z = sigmoid(rgb.z, p_shoulder_contrast, p_toe_contrast, p_general_contrast, log_midgray + p_pivot_offset, midgray);
      float3 img = rgb;
      // Linearize the formed image assuming its native transfer function is Rec.1886 curve
      img = spowf3(img, image_native_power);

      // lerp pre- and post-curve chromaticity angle
      float3 post_curve_hsv;
      rgb_to_hsv_v(img, post_curve_hsv);
      post_curve_hsv[0] = lerp_chromaticity_angle(pre_curve_hsv[0], post_curve_hsv[0], p_per_channel_hue_flight);
      hsv_to_rgb_v(post_curve_hsv, img);

      // generate outset matrix
      float3x3 outsetmat;
      if (p_use_inverse_inset) {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
            p_attenuation_rates.x, p_attenuation_rates.y, p_attenuation_rates.z, // Uses attenuation settings
            p_hue_flights.x, p_hue_flights.y, p_hue_flights.z,         // Uses attenuation settings
            p_tinting_rotate + 180, p_tinting_outset);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
      }
      else {
        Chromaticities outset_chromaticities = InsetPrimaries(
            COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
            p_restore_purity.x, p_restore_purity.y, p_restore_purity.z,
            p_reverse_hue_flights.x, p_reverse_hue_flights.y, p_reverse_hue_flights.z,
            p_tinting_rotate + 180, p_tinting_outset);
        outsetmat = inv_f33(RGBtoRGB(outset_chromaticities, COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)]));
      }

      // apply outset matrix
      img = mult_f3_f33(img, outsetmat);

      // convert from working primaries to target display primaries
      float3x3 working_to_display = RGBtoRGB(COLOR_SPACE_PRI[static_cast<int>(p_working_primaries)],
                                             COLOR_SPACE_PRI[static_cast<int>(p_display_primaries)]);
      img = mult_f3_f33(img, working_to_display);

      // apply low-side guard rail if the UI checkbox is true, otherwise hard clamp to 0
      if (p_compensate_negatives) {
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
static void cmp_node_agx_view_transform_build_multi_function(NodeMultiFunctionBuilder &builder)
{
  builder.set_instantiating_fn(+[](const bNode &node, mf::FunctionInitializer &initializer) {
    initializer.set_function<AgXViewTransformFunction>(node); // Pass the node to the constructor
  });
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
  ntype.initfunc = file_ns::cmp_node_agx_view_transform_init;
  ntype.draw_buttons = file_ns::cmp_node_agx_view_transform_layout;
  ntype.build_multi_function = file_ns::cmp_node_agx_view_transform_build_multi_function;
  blender::bke::node_type_storage(
      ntype, "NodeAgxViewTransform", node_free_agx_storage, node_copy_agx_storage);
  blender::bke::node_register_type(ntype);
  file_ns::cmp_node_agx_view_transform_rna(ntype.rna_ext.srna);
}
NOD_REGISTER_NODE(register_node_type_cmp_node_agx_view_transform)