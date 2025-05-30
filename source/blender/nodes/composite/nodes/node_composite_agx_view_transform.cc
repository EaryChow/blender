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
#include "RNA_access.hh"
#include "RNA_define.hh"
#include "RNA_enum_types.hh"
#include "UI_interface.hh"
#include "UI_resources.hh"


// Namespace Declaration
namespace blender::nodes::node_composite_agx_view_transform_cc {

// Node Declaration
static void node_declare(NodeDeclarationBuilder &b) {
  b.use_custom_socket_order();
  b.allow_any_socket_order();

  b.add_output<decl::Color>("Color");

  b.add_input<decl::Color>("Color")
      .default_value({1.0f, 1.0f, 1.0f, 1.0f})
      .compositor_domain_priority(0);

}

// Registration Function
static void node_register()
{
  static blender::bke::bNodeType ntype;

  cmp_node_type_base(&ntype, "CompositorNodeAgXViewTransform");
  ntype.ui_name = "AgX View Transform";
  ntype.ui_description = "Applies AgX Picture Formation that converts rendered RGB exposure into an Image for Display";
  ntype.nclass = NODE_CLASS_OP_COLOR;
  ntype.declare = node_declare;
  blender::bke::node_type_size_preset(ntype, blender::bke::eNodeSizePreset::Large);
  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(node_register)

}  // namespace blender::nodes::node_composite_agx_view_transform_cc
