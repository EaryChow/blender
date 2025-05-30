/* SPDX-FileCopyrightText: 2019 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */
#include "usd.hh"

#include "usd_armature_utils.hh"
#include "usd_blend_shape_utils.hh"
#include "usd_hierarchy_iterator.hh"
#include "usd_skel_convert.hh"
#include "usd_skel_root_utils.hh"
#include "usd_utils.hh"
#include "usd_writer_abstract.hh"
#include "usd_writer_armature.hh"
#include "usd_writer_camera.hh"
#include "usd_writer_curves.hh"
#include "usd_writer_hair.hh"
#include "usd_writer_light.hh"
#include "usd_writer_mesh.hh"
#include "usd_writer_metaball.hh"
#include "usd_writer_points.hh"
#include "usd_writer_text.hh"
#include "usd_writer_transform.hh"
#include "usd_writer_volume.hh"

#include <string>

#include "BKE_main.hh"

#include "BLI_assert.h"

#include "DNA_layer_types.h"
#include "DNA_object_types.h"

namespace blender::io::usd {

USDHierarchyIterator::USDHierarchyIterator(Main *bmain,
                                           Depsgraph *depsgraph,
                                           pxr::UsdStageRefPtr stage,
                                           const USDExportParams &params)
    : AbstractHierarchyIterator(bmain, depsgraph), stage_(stage), params_(params)
{
}

bool USDHierarchyIterator::mark_as_weak_export(const Object *object) const
{
  if (params_.selected_objects_only && (object->base_flag & BASE_SELECTED) == 0) {
    return true;
  }

  switch (object->type) {
    case OB_EMPTY:
      /* Always assume empties are being exported intentionally. */
      return false;
    case OB_MESH:
    case OB_MBALL:
    case OB_FONT:
      return !params_.export_meshes;
    case OB_CAMERA:
      return !params_.export_cameras;
    case OB_LAMP:
      return !params_.export_lights;
    case OB_CURVES_LEGACY:
    case OB_CURVES:
      return !params_.export_curves;
    case OB_VOLUME:
      return !params_.export_volumes;
    case OB_ARMATURE:
      return !params_.export_armatures;
    case OB_POINTCLOUD:
      return !params_.export_points;

    default:
      /* Assume weak for all other types. */
      return true;
  }
}

void USDHierarchyIterator::release_writer(AbstractHierarchyWriter *writer)
{
  delete static_cast<USDAbstractWriter *>(writer);
}

std::string USDHierarchyIterator::make_valid_name(const std::string &name) const
{
  return make_safe_name(name, params_.allow_unicode);
}

void USDHierarchyIterator::process_usd_skel() const
{
  skel_export_chaser(stage_,
                     armature_export_map_,
                     skinned_mesh_export_map_,
                     shape_key_mesh_export_map_,
                     depsgraph_);

  create_skel_roots(stage_, params_);
}

void USDHierarchyIterator::set_export_frame(float frame_nr)
{
  /* The USD stage is already set up to have FPS time-codes per frame. */
  export_time_ = pxr::UsdTimeCode(frame_nr);
}

USDExporterContext USDHierarchyIterator::create_usd_export_context(const HierarchyContext *context)
{
  pxr::SdfPath path;
  if (params_.root_prim_path[0] != '\0') {
    path = pxr::SdfPath(params_.root_prim_path + context->export_path);
  }
  else {
    path = pxr::SdfPath(context->export_path);
  }

  if (params_.merge_parent_xform && context->is_object_data_context && !context->is_parent) {
    bool can_merge_with_xform = true;
    if (params_.export_shapekeys && is_mesh_with_shape_keys(context->object)) {
      can_merge_with_xform = false;
    }

    if (params_.use_instancing && (context->is_prototype() || context->is_instance())) {
      can_merge_with_xform = false;
    }

    if (can_merge_with_xform) {
      path = path.GetParentPath();
    }
  }

  /* Returns the same path that was passed to `stage_` object during it's creation (via
   * `pxr::UsdStage::CreateNew` function). */
  const pxr::SdfLayerHandle root_layer = stage_->GetRootLayer();
  const std::string export_file_path = root_layer->GetRealPath();
  auto get_time_code = [this]() { return this->export_time_; };

  return USDExporterContext{
      bmain_, depsgraph_, stage_, path, get_time_code, params_, export_file_path};
}

AbstractHierarchyWriter *USDHierarchyIterator::create_transform_writer(
    const HierarchyContext *context)
{
  return new USDTransformWriter(create_usd_export_context(context));
}

AbstractHierarchyWriter *USDHierarchyIterator::create_data_writer(const HierarchyContext *context)
{
  USDExporterContext usd_export_context = create_usd_export_context(context);
  USDAbstractWriter *data_writer = nullptr;

  switch (context->object->type) {
    case OB_MESH:
      if (usd_export_context.export_params.export_meshes) {
        data_writer = new USDMeshWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_CAMERA:
      if (usd_export_context.export_params.export_cameras) {
        data_writer = new USDCameraWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_LAMP:
      if (usd_export_context.export_params.export_lights) {
        data_writer = new USDLightWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_MBALL:
      data_writer = new USDMetaballWriter(usd_export_context);
      break;
    case OB_FONT:
      data_writer = new USDTextWriter(usd_export_context);
      break;
    case OB_CURVES_LEGACY:
    case OB_CURVES:
      if (usd_export_context.export_params.export_curves) {
        data_writer = new USDCurvesWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_VOLUME:
      if (usd_export_context.export_params.export_volumes) {
        data_writer = new USDVolumeWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_ARMATURE:
      if (usd_export_context.export_params.export_armatures) {
        data_writer = new USDArmatureWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;
    case OB_POINTCLOUD:
      if (usd_export_context.export_params.export_points) {
        data_writer = new USDPointsWriter(usd_export_context);
      }
      else {
        return nullptr;
      }
      break;

    case OB_EMPTY:
    case OB_SURF:
    case OB_SPEAKER:
    case OB_LIGHTPROBE:
    case OB_LATTICE:
    case OB_GREASE_PENCIL:
      return nullptr;

    case OB_TYPE_MAX:
      BLI_assert_msg(0, "OB_TYPE_MAX should not be used");
      return nullptr;
    default:
      BLI_assert_unreachable();
      return nullptr;
  }

  if (!data_writer->is_supported(context)) {
    delete data_writer;
    return nullptr;
  }

  if (data_writer && (params_.export_armatures || params_.export_shapekeys)) {
    add_usd_skel_export_mapping(context->object, data_writer->usd_path());
  }

  return data_writer;
}

AbstractHierarchyWriter *USDHierarchyIterator::create_hair_writer(const HierarchyContext *context)
{
  if (!params_.export_hair) {
    return nullptr;
  }
  return new USDHairWriter(create_usd_export_context(context));
}

AbstractHierarchyWriter *USDHierarchyIterator::create_particle_writer(
    const HierarchyContext * /*context*/)
{
  return nullptr;
}

bool USDHierarchyIterator::include_data_writers(const HierarchyContext *context) const
{
  /* Don't generate data writers for instances. */

  return !(params_.use_instancing && context->is_instance());
}

bool USDHierarchyIterator::include_child_writers(const HierarchyContext *context) const
{
  /* Don't generate writers for children of instances. */

  return !(params_.use_instancing && context->is_instance());
}

void USDHierarchyIterator::add_usd_skel_export_mapping(const Object *obj, const pxr::SdfPath &path)
{
  if (params_.export_shapekeys && is_mesh_with_shape_keys(obj)) {
    shape_key_mesh_export_map_.add(obj, path);
  }

  if (params_.export_armatures && obj->type == OB_ARMATURE) {
    armature_export_map_.add(obj, path);
  }

  if (params_.export_armatures && obj->type == OB_MESH &&
      can_export_skinned_mesh(*obj, depsgraph_))
  {
    skinned_mesh_export_map_.add(obj, path);
  }
}

}  // namespace blender::io::usd
