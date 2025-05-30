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

  blender::bke::node_register_type(ntype);
}
NOD_REGISTER_NODE(register_node_type_cmp_brightcontrast)
