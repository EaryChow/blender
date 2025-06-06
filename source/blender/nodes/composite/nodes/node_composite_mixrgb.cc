/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup cmpnodes
 */

#include "BLI_assert.h"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_types.hh"

#include "FN_multi_function_builder.hh"

#include "DNA_material_types.h"

#include "BKE_material.hh"

#include "GPU_material.hh"

#include "COM_utilities_gpu_material.hh"

#include "NOD_multi_function.hh"
#include "NOD_socket_search_link.hh"

#include "RNA_enum_types.hh"

#include "node_composite_util.hh"

/* **************** MIX RGB ******************** */

namespace blender::nodes::node_composite_mixrgb_cc {

static void cmp_node_mixrgb_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Float>("Fac")
      .default_value(1.0f)
      .min(0.0f)
      .max(1.0f)
      .subtype(PROP_FACTOR)
      .compositor_domain_priority(2);
  b.add_input<decl::Color>("Image")
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);
  b.add_input<decl::Color>("Image", "Image_001")
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(1);
  b.add_output<decl::Color>("Image");
}

using namespace blender::compositor;

static int get_mode(const bNode &node)
{
  return node.custom1;
}

static bool get_use_alpha(const bNode &node)
{
  return node.custom2 & SHD_MIXRGB_USE_ALPHA;
}

static bool get_should_clamp(const bNode &node)
{
  return node.custom2 & SHD_MIXRGB_CLAMP;
}

static const char *get_shader_function_name(const bNode &node)
{
  switch (get_mode(node)) {
    case MA_RAMP_BLEND:
      return "mix_blend";
    case MA_RAMP_ADD:
      return "mix_add";
    case MA_RAMP_MULT:
      return "mix_mult";
    case MA_RAMP_SUB:
      return "mix_sub";
    case MA_RAMP_SCREEN:
      return "mix_screen";
    case MA_RAMP_DIV:
      return "mix_div";
    case MA_RAMP_DIFF:
      return "mix_diff";
    case MA_RAMP_EXCLUSION:
      return "mix_exclusion";
    case MA_RAMP_DARK:
      return "mix_dark";
    case MA_RAMP_LIGHT:
      return "mix_light";
    case MA_RAMP_OVERLAY:
      return "mix_overlay";
    case MA_RAMP_DODGE:
      return "mix_dodge";
    case MA_RAMP_BURN:
      return "mix_burn";
    case MA_RAMP_HUE:
      return "mix_hue";
    case MA_RAMP_SAT:
      return "mix_sat";
    case MA_RAMP_VAL:
      return "mix_val";
    case MA_RAMP_COLOR:
      return "mix_color";
    case MA_RAMP_SOFT:
      return "mix_soft";
    case MA_RAMP_LINEAR:
      return "mix_linear";
  }

  BLI_assert_unreachable();
  return nullptr;
}

static int node_gpu_material(GPUMaterial *material,
                             bNode *node,
                             bNodeExecData * /*execdata*/,
                             GPUNodeStack *inputs,
                             GPUNodeStack *outputs)
{
  if (get_use_alpha(*node)) {
    GPU_link(material,
             "multiply_by_alpha",
             get_shader_node_input_link(*node, inputs, "Fac"),
             get_shader_node_input_link(*node, inputs, "Image_001"),
             &get_shader_node_input(*node, inputs, "Fac").link);
  }

  const bool is_valid = GPU_stack_link(
      material, node, get_shader_function_name(*node), inputs, outputs);

  if (!is_valid || !get_should_clamp(*node)) {
    return is_valid;
  }

  const float min[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float max[4] = {1.0f, 1.0f, 1.0f, 1.0f};
  return GPU_link(material,
                  "clamp_color",
                  get_shader_node_output(*node, outputs, "Image").link,
                  GPU_constant(min),
                  GPU_constant(max),
                  &get_shader_node_output(*node, outputs, "Image").link);
}

static void node_build_multi_function(blender::nodes::NodeMultiFunctionBuilder &builder)
{
  const int mode = get_mode(builder.node());

  if (get_use_alpha(builder.node())) {
    if (get_should_clamp(builder.node())) {
      builder.construct_and_set_matching_fn_cb([=]() {
        return mf::build::SI3_SO<float, float4, float4, float4>(
            "Mix RGB Alpha Clamp",
            [=](const float factor, const float4 &color1, const float4 &color2) -> float4 {
              const float alpha_factor = factor * color2.w;
              float4 result = color1;
              ramp_blend(mode, result, alpha_factor, color2);
              return math::clamp(result, 0.0f, 1.0f);
            },
            mf::build::exec_presets::SomeSpanOrSingle<1, 2>());
      });
    }
    else {
      builder.construct_and_set_matching_fn_cb([=]() {
        return mf::build::SI3_SO<float, float4, float4, float4>(
            "Mix RGB Alpha",
            [=](const float factor, const float4 &color1, const float4 &color2) -> float4 {
              const float alpha_factor = factor * color2.w;
              float4 result = color1;
              ramp_blend(mode, result, alpha_factor, color2);
              return result;
            },
            mf::build::exec_presets::SomeSpanOrSingle<1, 2>());
      });
    }
  }
  else {
    if (get_should_clamp(builder.node())) {
      builder.construct_and_set_matching_fn_cb([=]() {
        return mf::build::SI3_SO<float, float4, float4, float4>(
            "Mix RGB Clamp",
            [=](const float factor, const float4 &color1, const float4 &color2) -> float4 {
              float4 result = color1;
              ramp_blend(mode, result, factor, color2);
              return math::clamp(result, 0.0f, 1.0f);
            },
            mf::build::exec_presets::SomeSpanOrSingle<1, 2>());
      });
    }
    else {
      builder.construct_and_set_matching_fn_cb([=]() {
        return mf::build::SI3_SO<float, float4, float4, float4>(
            "Mix RGB",
            [=](const float factor, const float4 &color1, const float4 &color2) -> float4 {
              float4 result = color1;
              ramp_blend(mode, result, factor, color2);
              return result;
            },
            mf::build::exec_presets::SomeSpanOrSingle<1, 2>());
      });
    }
  }
}

}  // namespace blender::nodes::node_composite_mixrgb_cc

static void register_node_type_cmp_mix_rgb()
{
  namespace file_ns = blender::nodes::node_composite_mixrgb_cc;

  static blender::bke::bNodeType ntype;

  cmp_node_type_base(&ntype, "CompositorNodeMixRGB", CMP_NODE_MIX_RGB);
  ntype.ui_name = "Mix";
  ntype.ui_description = "Blend two images together using various blending modes";
  ntype.enum_name_legacy = "MIX_RGB";
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.flag |= NODE_PREVIEW;
  ntype.declare = file_ns::cmp_node_mixrgb_declare;
  ntype.labelfunc = node_blend_label;
  ntype.gpu_fn = file_ns::node_gpu_material;
  ntype.build_multi_function = file_ns::node_build_multi_function;
  ntype.gather_link_search_ops = nullptr;

  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(register_node_type_cmp_mix_rgb)
