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
    {int(AGXWorkingLog::AGX_WORKING_LOG_LINEAR), "linear", 0, "Linear", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_ACESCCT), "acescct", 0, "ACEScct", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_ARRI_LOGC3), "arri_logc3", 0, "ARRI LogC3", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_ARRI_LOGC4), "arri_logc4", 0, "ARRI LogC4", ""},
    {int(AGXWorkingLog::AGX_WORKING_LOG_GENERIC_LOG2), "generic_log2", 0, "Generic Log2", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

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
