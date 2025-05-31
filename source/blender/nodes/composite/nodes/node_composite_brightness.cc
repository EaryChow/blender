/* SPDX-FileCopyrightText: 2006 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup cmpnodes
 */

#include <limits>

#include "BLI_math_color.h"
#include "BLI_math_vector_types.hh"

#include "FN_multi_function_builder.hh"

#include "NOD_multi_function.hh"

#include "UI_interface.hh"
#include "UI_resources.hh"

#include "GPU_material.hh"

#include "node_composite_util.hh"

/* **************** Brightness and Contrast  ******************** */

namespace blender::nodes::node_composite_brightness_cc {

static void cmp_node_brightcontrast_declare(NodeDeclarationBuilder &b)
{
  b.add_input<decl::Color>("Image").default_value({1.0f, 1.0f, 1.0f, 1.0f});
  b.add_input<decl::Float>("Bright").min(-100.0f).max(100.0f);
  b.add_input<decl::Float>("Contrast").min(-100.0f).max(100.0f);
  b.add_output<decl::Color>("Image");
}

using namespace blender::compositor;

static int node_gpu_material(GPUMaterial *material,
                             bNode *node,
                             bNodeExecData * /*execdata*/,
                             GPUNodeStack *inputs,
                             GPUNodeStack *outputs)
{
  return GPU_stack_link(material, node, "node_composite_bright_contrast", inputs, outputs);
}

// Debug test: replace Multi-function Builder with AgX's code
class brightness_and_contrast_Function : public mf::MultiFunction {
 public:
  brightness_and_contrast_Function(const bNode &node) {
    static const mf::Signature signature = []() {
      mf::Signature signature;
      mf::SignatureBuilder builder{"brightness_and_contrast", signature};
      // Socket Inputs:
      builder.single_input<float4>("In Color");                         // Index 0
      builder.single_input<float>("Bright");                  // Index 1
      builder.single_input<float>("Contrast");                      // Index 2
      // Output:
      builder.single_output<float4>("Out Color");                        // Index 15
      return signature;
    }();
    this->set_signature(&signature);
  }

  void call(const IndexMask &mask, mf::Params params, mf::Context /*context*/) const override {
    const VArray<float4> in_color = params.readonly_single_input<float4>(0, "In Color");
    const VArray<float> general_contrast_in = params.readonly_single_input<float>(1, "Bright");
    const VArray<float> toe_contrast_in = params.readonly_single_input<float>(2, "Contrast");
    MutableSpan<float4> out_color = params.uninitialized_single_output<float4>(3, "Out Color");

    mask.foreach_index([&](const int64_t i) {
      float4 col = in_color[i];
      out_color[i] = col;
    });
  }
};

static void node_build_multi_function(NodeMultiFunctionBuilder &builder) {
  builder.construct_and_set_matching_fn<brightness_and_contrast_Function>(builder.node());
}


}  // namespace blender::nodes::node_composite_brightness_cc

static void register_node_type_cmp_brightcontrast()
{
  namespace file_ns = blender::nodes::node_composite_brightness_cc;

  static blender::bke::bNodeType ntype;

  cmp_node_type_base(&ntype, "CompositorNodeBrightContrast", CMP_NODE_BRIGHTCONTRAST);
  ntype.ui_name = "Brightness/Contrast";
  ntype.ui_description = "Adjust brightness and contrast";
  ntype.enum_name_legacy = "BRIGHTCONTRAST";
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.declare = file_ns::cmp_node_brightcontrast_declare;
  ntype.gpu_fn = file_ns::node_gpu_material;
  ntype.build_multi_function = file_ns::node_build_multi_function;

  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(register_node_type_cmp_brightcontrast)
