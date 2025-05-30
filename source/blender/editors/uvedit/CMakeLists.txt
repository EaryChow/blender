# SPDX-FileCopyrightText: 2023 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

set(INC
  ../include
  ../../makesrna
  ../../../../intern/eigen
  # RNA_prototypes.hh
  ${CMAKE_BINARY_DIR}/source/blender/makesrna
)

set(INC_SYS
)

set(SRC
  uvedit_buttons.cc
  uvedit_clipboard.cc
  uvedit_clipboard_graph_iso.cc
  uvedit_draw.cc
  uvedit_islands.cc
  uvedit_ops.cc
  uvedit_path.cc
  uvedit_rip.cc
  uvedit_select.cc
  uvedit_smart_stitch.cc
  uvedit_unwrap_ops.cc

  uvedit_clipboard_graph_iso.hh
  uvedit_intern.hh
)

set(LIB
  PRIVATE bf::blenkernel
  PRIVATE bf::blenlib
  PRIVATE bf::blentranslation
  PRIVATE bf::bmesh
  PRIVATE bf::depsgraph
  PRIVATE bf::dna
  PRIVATE bf::extern::fmtlib
  PRIVATE bf::geometry
  PRIVATE bf::gpu
  PRIVATE bf::intern::guardedalloc
  PRIVATE bf::nodes
  PRIVATE bf::windowmanager
)

if(WITH_UV_SLIM)
  list(APPEND LIB
    bf_intern_slim
  )
  add_definitions(-DWITH_UV_SLIM)
endif()


blender_add_lib(bf_editor_uvedit "${SRC}" "${INC}" "${INC_SYS}" "${LIB}")

# RNA_prototypes.hh
add_dependencies(bf_editor_uvedit bf_rna)
