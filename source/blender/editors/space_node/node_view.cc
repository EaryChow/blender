/* SPDX-FileCopyrightText: 2008 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup spnode
 */

#include "DNA_node_types.h"

#include "BLI_rect.h"
#include "BLI_utildefines.h"

#include "BKE_context.hh"
#include "BKE_image.hh"
#include "BKE_node_runtime.hh"
#include "BKE_screen.hh"

#include "ED_image.hh"
#include "ED_node.hh" /* own include */
#include "ED_screen.hh"
#include "ED_space_api.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "UI_view2d.hh"

#include "MEM_guardedalloc.h"

#include "IMB_colormanagement.hh"
#include "IMB_imbuf.hh"
#include "IMB_imbuf_types.hh"

#include "node_intern.hh" /* own include */

namespace blender::ed::space_node {

/* -------------------------------------------------------------------- */
/** \name Local Functions
 * \{ */

static bool space_node_active_view_poll(bContext *C)
{
  if (!ED_operator_node_active(C)) {
    return false;
  }
  const ARegion *region = CTX_wm_region(C);
  if (!(region && region->regiontype == RGN_TYPE_WINDOW)) {
    return false;
  }
  return true;
}

static bool space_node_composite_active_view_poll(bContext *C)
{
  if (!composite_node_active(C)) {
    return false;
  }
  const ARegion *region = CTX_wm_region(C);
  if (!(region && region->regiontype == RGN_TYPE_WINDOW)) {
    return false;
  }
  return true;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name View All Operator
 * \{ */

bool space_node_view_flag(
    bContext &C, SpaceNode &snode, ARegion &region, const int node_flag, const int smooth_viewtx)
{
  const float oldwidth = BLI_rctf_size_x(&region.v2d.cur);
  const float oldheight = BLI_rctf_size_y(&region.v2d.cur);

  const float old_aspect = oldwidth / oldheight;

  rctf cur_new;
  BLI_rctf_init_minmax(&cur_new);

  int tot = 0;
  bool has_frame = false;
  if (snode.edittree) {
    for (const bNode *node : snode.edittree->all_nodes()) {
      if ((node->flag & node_flag) == node_flag) {
        BLI_rctf_union(&cur_new, &node->runtime->draw_bounds);
        tot++;

        if (node->is_frame()) {
          has_frame = true;
        }
      }
    }
  }

  if (tot == 0) {
    return false;
  }

  const float width = BLI_rctf_size_x(&cur_new);
  const float height = BLI_rctf_size_y(&cur_new);
  const float new_aspect = width / height;

  /* for single non-frame nodes, don't zoom in, just pan view,
   * but do allow zooming out, this allows for big nodes to be zoomed out */
  if ((tot == 1) && (has_frame == false) && ((oldwidth * oldheight) > (width * height))) {
    /* center, don't zoom */
    BLI_rctf_resize(&cur_new, oldwidth, oldheight);
  }
  else {
    if (old_aspect < new_aspect) {
      const float height_new = width / old_aspect;
      cur_new.ymin = cur_new.ymin - height_new / 2.0f;
      cur_new.ymax = cur_new.ymax + height_new / 2.0f;
    }
    else {
      const float width_new = height * old_aspect;
      cur_new.xmin = cur_new.xmin - width_new / 2.0f;
      cur_new.xmax = cur_new.xmax + width_new / 2.0f;
    }

    /* add some padding */
    BLI_rctf_scale(&cur_new, 1.1f);
  }

  UI_view2d_smooth_view(&C, &region, &cur_new, smooth_viewtx);

  return true;
}

static wmOperatorStatus node_view_all_exec(bContext *C, wmOperator *op)
{
  ARegion *region = CTX_wm_region(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  const int smooth_viewtx = WM_operator_smooth_viewtx_get(op);

  /* is this really needed? */
  snode->xof = 0;
  snode->yof = 0;

  if (space_node_view_flag(*C, *snode, *region, 0, smooth_viewtx)) {
    return OPERATOR_FINISHED;
  }
  return OPERATOR_CANCELLED;
}

void NODE_OT_view_all(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Frame All";
  ot->idname = "NODE_OT_view_all";
  ot->description = "Resize view so you can see all nodes";

  /* API callbacks. */
  ot->exec = node_view_all_exec;
  ot->poll = space_node_active_view_poll;

  /* flags */
  ot->flag = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name View Selected Operator
 * \{ */

static wmOperatorStatus node_view_selected_exec(bContext *C, wmOperator *op)
{
  ARegion *region = CTX_wm_region(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  const int smooth_viewtx = WM_operator_smooth_viewtx_get(op);

  if (space_node_view_flag(*C, *snode, *region, NODE_SELECT, smooth_viewtx)) {
    return OPERATOR_FINISHED;
  }
  return OPERATOR_CANCELLED;
}

void NODE_OT_view_selected(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Frame Selected";
  ot->idname = "NODE_OT_view_selected";
  ot->description = "Resize view so you can see selected nodes";

  /* API callbacks. */
  ot->exec = node_view_selected_exec;
  ot->poll = space_node_active_view_poll;

  /* flags */
  ot->flag = 0;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Operators
 * \{ */

struct NodeViewMove {
  int2 mvalo;
  int xmin, ymin, xmax, ymax;
  /** Original Offset for cancel. */
  float xof_orig, yof_orig;
};

static wmOperatorStatus snode_bg_viewmove_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  NodeViewMove *nvm = (NodeViewMove *)op->customdata;

  switch (event->type) {
    case MOUSEMOVE:

      snode->xof -= (nvm->mvalo.x - event->mval[0]);
      snode->yof -= (nvm->mvalo.y - event->mval[1]);
      nvm->mvalo.x = event->mval[0];
      nvm->mvalo.y = event->mval[1];

      /* prevent dragging image outside of the window and losing it! */
      CLAMP(snode->xof, nvm->xmin, nvm->xmax);
      CLAMP(snode->yof, nvm->ymin, nvm->ymax);

      ED_region_tag_redraw(region);
      WM_main_add_notifier(NC_NODE | ND_DISPLAY, nullptr);
      WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, nullptr);

      break;

    case LEFTMOUSE:
    case MIDDLEMOUSE:
      if (event->val == KM_RELEASE) {
        MEM_freeN(nvm);
        op->customdata = nullptr;
        return OPERATOR_FINISHED;
      }
      break;
    case EVT_ESCKEY:
    case RIGHTMOUSE:
      snode->xof = nvm->xof_orig;
      snode->yof = nvm->yof_orig;
      ED_region_tag_redraw(region);
      WM_main_add_notifier(NC_NODE | ND_DISPLAY, nullptr);
      WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, nullptr);

      MEM_freeN(nvm);
      op->customdata = nullptr;

      return OPERATOR_CANCELLED;
    default: {
      break;
    }
  }

  return OPERATOR_RUNNING_MODAL;
}

static wmOperatorStatus snode_bg_viewmove_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  NodeViewMove *nvm;
  Image *ima;
  ImBuf *ibuf;
  const float pad = 32.0f; /* Better be bigger than scroll-bars. */

  void *lock;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, nullptr, &lock);

  if (ibuf == nullptr) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return OPERATOR_CANCELLED;
  }

  nvm = MEM_callocN<NodeViewMove>(__func__);
  op->customdata = nvm;
  nvm->mvalo.x = event->mval[0];
  nvm->mvalo.y = event->mval[1];

  nvm->xmin = -(region->winx / 2) - (ibuf->x * (0.5f * snode->zoom)) + pad;
  nvm->xmax = (region->winx / 2) + (ibuf->x * (0.5f * snode->zoom)) - pad;
  nvm->ymin = -(region->winy / 2) - (ibuf->y * (0.5f * snode->zoom)) + pad;
  nvm->ymax = (region->winy / 2) + (ibuf->y * (0.5f * snode->zoom)) - pad;

  nvm->xof_orig = snode->xof;
  nvm->yof_orig = snode->yof;

  BKE_image_release_ibuf(ima, ibuf, lock);

  /* add modal handler */
  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static void snode_bg_viewmove_cancel(bContext * /*C*/, wmOperator *op)
{
  NodeViewMove *nvm = (NodeViewMove *)op->customdata;
  MEM_freeN(nvm);
  op->customdata = nullptr;
}

void NODE_OT_backimage_move(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Background Image Move";
  ot->description = "Move node backdrop";
  ot->idname = "NODE_OT_backimage_move";

  /* API callbacks. */
  ot->invoke = snode_bg_viewmove_invoke;
  ot->modal = snode_bg_viewmove_modal;
  ot->poll = space_node_composite_active_view_poll;
  ot->cancel = snode_bg_viewmove_cancel;

  /* flags */
  ot->flag = OPTYPE_BLOCKING | OPTYPE_GRAB_CURSOR_XY;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Zoom
 * \{ */

static wmOperatorStatus backimage_zoom_exec(bContext *C, wmOperator *op)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  float fac = RNA_float_get(op->ptr, "factor");

  snode->zoom *= fac;
  ED_region_tag_redraw(region);
  WM_main_add_notifier(NC_NODE | ND_DISPLAY, nullptr);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, nullptr);

  return OPERATOR_FINISHED;
}

void NODE_OT_backimage_zoom(wmOperatorType *ot)
{

  /* identifiers */
  ot->name = "Background Image Zoom";
  ot->idname = "NODE_OT_backimage_zoom";
  ot->description = "Zoom in/out the background image";

  /* API callbacks. */
  ot->exec = backimage_zoom_exec;
  ot->poll = space_node_composite_active_view_poll;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;

  /* internal */
  RNA_def_float(ot->srna, "factor", 1.2f, 0.0f, 10.0f, "Factor", "", 0.0f, 10.0f);
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Background Image Fit
 * \{ */

static wmOperatorStatus backimage_fit_exec(bContext *C, wmOperator * /*op*/)
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);

  Image *ima;
  ImBuf *ibuf;

  const float pad = 32.0f;

  void *lock;

  float facx, facy;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, nullptr, &lock);

  if ((ibuf == nullptr) || (ibuf->x == 0) || (ibuf->y == 0)) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return OPERATOR_CANCELLED;
  }

  facx = 1.0f * (region->sizex - pad) / (ibuf->x * snode->zoom);
  facy = 1.0f * (region->sizey - pad) / (ibuf->y * snode->zoom);

  BKE_image_release_ibuf(ima, ibuf, lock);

  snode->zoom *= min_ff(facx, facy) * UI_SCALE_FAC;

  snode->xof = 0;
  snode->yof = 0;

  ED_region_tag_redraw(region);
  WM_main_add_notifier(NC_NODE | ND_DISPLAY, nullptr);
  WM_main_add_notifier(NC_SPACE | ND_SPACE_NODE_VIEW, nullptr);

  return OPERATOR_FINISHED;
}

void NODE_OT_backimage_fit(wmOperatorType *ot)
{

  /* identifiers */
  ot->name = "Background Image Fit";
  ot->idname = "NODE_OT_backimage_fit";
  ot->description = "Fit the background image to the view";

  /* API callbacks. */
  ot->exec = backimage_fit_exec;
  ot->poll = space_node_composite_active_view_poll;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;
}

/** \} */

/* -------------------------------------------------------------------- */
/** \name Sample Backdrop Operator
 * \{ */

struct ImageSampleInfo {
  ARegionType *art;
  void *draw_handle;
  int x, y;
  int channels;

  uchar col[4];
  float colf[4];
  float linearcol[4];

  int draw;
  int color_manage;
};

static void sample_draw(const bContext *C, ARegion *region, void *arg_info)
{
  Scene *scene = CTX_data_scene(C);
  ImageSampleInfo *info = (ImageSampleInfo *)arg_info;

  if (info->draw) {
    ED_image_draw_info(scene,
                       region,
                       info->color_manage,
                       false,
                       info->channels,
                       info->x,
                       info->y,
                       info->col,
                       info->colf,
                       info->linearcol);
  }
}

}  // namespace blender::ed::space_node

bool ED_space_node_get_position(
    Main *bmain, SpaceNode *snode, ARegion *region, const int mval[2], float fpos[2])
{
  if (!ED_node_is_compositor(snode) || (snode->flag & SNODE_BACKDRAW) == 0) {
    return false;
  }

  void *lock;
  Image *ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ImBuf *ibuf = BKE_image_acquire_ibuf(ima, nullptr, &lock);
  if (!ibuf) {
    BKE_image_release_ibuf(ima, ibuf, lock);
    return false;
  }

  /* map the mouse coords to the backdrop image space */
  float bufx = ibuf->x * snode->zoom;
  float bufy = ibuf->y * snode->zoom;
  fpos[0] = (bufx > 0.0f ? (float(mval[0]) - 0.5f * region->winx - snode->xof) / bufx + 0.5f :
                           0.0f);
  fpos[1] = (bufy > 0.0f ? (float(mval[1]) - 0.5f * region->winy - snode->yof) / bufy + 0.5f :
                           0.0f);

  BKE_image_release_ibuf(ima, ibuf, lock);
  return true;
}

bool ED_space_node_color_sample(
    Main *bmain, SpaceNode *snode, ARegion *region, const int mval[2], float r_col[3])
{
  void *lock;
  Image *ima;
  ImBuf *ibuf;
  float fx, fy, bufx, bufy;
  bool ret = false;

  if (!ED_node_is_compositor(snode) || (snode->flag & SNODE_BACKDRAW) == 0) {
    /* use viewer image for color sampling only if we're in compositor tree
     * with backdrop enabled
     */
    return false;
  }

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, nullptr, &lock);
  if (!ibuf) {
    return false;
  }

  /* map the mouse coords to the backdrop image space */
  bufx = ibuf->x * snode->zoom;
  bufy = ibuf->y * snode->zoom;
  fx = (bufx > 0.0f ? (float(mval[0]) - 0.5f * region->winx - snode->xof) / bufx + 0.5f : 0.0f);
  fy = (bufy > 0.0f ? (float(mval[1]) - 0.5f * region->winy - snode->yof) / bufy + 0.5f : 0.0f);

  if (fx >= 0.0f && fy >= 0.0f && fx < 1.0f && fy < 1.0f) {
    const float *fp;
    uchar *cp;
    int x = int(fx * ibuf->x), y = int(fy * ibuf->y);

    CLAMP(x, 0, ibuf->x - 1);
    CLAMP(y, 0, ibuf->y - 1);

    if (ibuf->float_buffer.data) {
      fp = (ibuf->float_buffer.data + (ibuf->channels) * (y * ibuf->x + x));
      /* #IB_PROFILE_NONE is default but in fact its linear. */
      copy_v3_v3(r_col, fp);
      ret = true;
    }
    else if (ibuf->byte_buffer.data) {
      cp = ibuf->byte_buffer.data + 4 * (y * ibuf->x + x);
      rgb_uchar_to_float(r_col, cp);
      IMB_colormanagement_colorspace_to_scene_linear_v3(r_col, ibuf->byte_buffer.colorspace);
      ret = true;
    }
  }

  BKE_image_release_ibuf(ima, ibuf, lock);

  return ret;
}

namespace blender::ed::space_node {

static void sample_apply(bContext *C, wmOperator *op, const wmEvent *event)
{
  Main *bmain = CTX_data_main(C);
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  ImageSampleInfo *info = (ImageSampleInfo *)op->customdata;
  void *lock;
  Image *ima;
  ImBuf *ibuf;
  float fx, fy, bufx, bufy;

  ima = BKE_image_ensure_viewer(bmain, IMA_TYPE_COMPOSITE, "Viewer Node");
  ibuf = BKE_image_acquire_ibuf(ima, nullptr, &lock);
  if (!ibuf) {
    info->draw = 0;
    return;
  }

  if (!ibuf->byte_buffer.data) {
    IMB_byte_from_float(ibuf);
  }

  /* map the mouse coords to the backdrop image space */
  bufx = ibuf->x * snode->zoom;
  bufy = ibuf->y * snode->zoom;
  fx = (bufx > 0.0f ? (float(event->mval[0]) - 0.5f * region->winx - snode->xof) / bufx + 0.5f :
                      0.0f);
  fy = (bufy > 0.0f ? (float(event->mval[1]) - 0.5f * region->winy - snode->yof) / bufy + 0.5f :
                      0.0f);

  if (fx >= 0.0f && fy >= 0.0f && fx < 1.0f && fy < 1.0f) {
    const float *fp;
    uchar *cp;
    int x = int(fx * ibuf->x), y = int(fy * ibuf->y);

    CLAMP(x, 0, ibuf->x - 1);
    CLAMP(y, 0, ibuf->y - 1);

    info->x = x;
    info->y = y;
    info->draw = 1;
    info->channels = ibuf->channels;

    if (ibuf->byte_buffer.data) {
      cp = ibuf->byte_buffer.data + 4 * (y * ibuf->x + x);

      info->col[0] = cp[0];
      info->col[1] = cp[1];
      info->col[2] = cp[2];
      info->col[3] = cp[3];

      info->colf[0] = float(cp[0]) / 255.0f;
      info->colf[1] = float(cp[1]) / 255.0f;
      info->colf[2] = float(cp[2]) / 255.0f;
      info->colf[3] = float(cp[3]) / 255.0f;

      copy_v4_v4(info->linearcol, info->colf);
      IMB_colormanagement_colorspace_to_scene_linear_v4(
          info->linearcol, false, ibuf->byte_buffer.colorspace);

      info->color_manage = true;
    }
    if (ibuf->float_buffer.data) {
      fp = (ibuf->float_buffer.data + (ibuf->channels) * (y * ibuf->x + x));

      info->colf[0] = fp[0];
      info->colf[1] = fp[1];
      info->colf[2] = fp[2];
      info->colf[3] = fp[3];

      info->color_manage = true;
    }

    ED_node_sample_set(info->colf);
  }
  else {
    info->draw = 0;
    ED_node_sample_set(nullptr);
  }

  BKE_image_release_ibuf(ima, ibuf, lock);

  ED_area_tag_redraw(CTX_wm_area(C));
}

static void sample_exit(bContext *C, wmOperator *op)
{
  ImageSampleInfo *info = (ImageSampleInfo *)op->customdata;

  ED_node_sample_set(nullptr);
  ED_region_draw_cb_exit(info->art, info->draw_handle);
  ED_area_tag_redraw(CTX_wm_area(C));
  MEM_freeN(info);
}

static wmOperatorStatus sample_invoke(bContext *C, wmOperator *op, const wmEvent *event)
{
  SpaceNode *snode = CTX_wm_space_node(C);
  ARegion *region = CTX_wm_region(C);
  ImageSampleInfo *info;

  /* Don't handle events intended for nodes (which rely on click/drag distinction).
   * which this operator would use since sampling is normally activated on press, see: #98191. */
  if (node_or_socket_isect_event(*C, *event)) {
    return OPERATOR_PASS_THROUGH;
  }

  if (!ED_node_is_compositor(snode) || !(snode->flag & SNODE_BACKDRAW)) {
    return OPERATOR_CANCELLED;
  }

  info = MEM_callocN<ImageSampleInfo>("ImageSampleInfo");
  info->art = region->runtime->type;
  info->draw_handle = ED_region_draw_cb_activate(
      region->runtime->type, sample_draw, info, REGION_DRAW_POST_PIXEL);
  op->customdata = info;

  sample_apply(C, op, event);

  WM_event_add_modal_handler(C, op);

  return OPERATOR_RUNNING_MODAL;
}

static wmOperatorStatus sample_modal(bContext *C, wmOperator *op, const wmEvent *event)
{
  switch (event->type) {
    case LEFTMOUSE:
    case RIGHTMOUSE: /* XXX hardcoded */
      if (event->val == KM_RELEASE) {
        sample_exit(C, op);
        return OPERATOR_CANCELLED;
      }
      break;
    case MOUSEMOVE:
      sample_apply(C, op, event);
      break;
    default: {
      break;
    }
  }

  return OPERATOR_RUNNING_MODAL;
}

static void sample_cancel(bContext *C, wmOperator *op)
{
  sample_exit(C, op);
}

void NODE_OT_backimage_sample(wmOperatorType *ot)
{
  /* identifiers */
  ot->name = "Backimage Sample";
  ot->idname = "NODE_OT_backimage_sample";
  ot->description = "Use mouse to sample background image";

  /* API callbacks. */
  ot->invoke = sample_invoke;
  ot->modal = sample_modal;
  ot->cancel = sample_cancel;
  ot->poll = space_node_active_view_poll;

  /* flags */
  ot->flag = OPTYPE_BLOCKING;
}

/** \} */

}  // namespace blender::ed::space_node
