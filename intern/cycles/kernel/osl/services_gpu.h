/* SPDX-FileCopyrightText: 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved.
 * SPDX-FileCopyrightText: 2011-2022 Blender Foundation
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted code from Open Shading Language. */

#include "kernel/tables.h"

#include "kernel/camera/camera.h"

#include "kernel/geom/attribute.h"
#include "kernel/geom/curve.h"
#include "kernel/geom/motion_triangle.h"
#include "kernel/geom/object.h"
#include "kernel/geom/point.h"
#include "kernel/geom/primitive.h"
#include "kernel/geom/triangle.h"

#include "kernel/svm/math_util.h"
#include "kernel/svm/noise.h"

#include "kernel/util/colorspace.h"
#include "kernel/util/differential.h"
#include "kernel/util/ies.h"

#include "util/hash.h"
#include "util/transform.h"

#include "kernel/osl/osl.h"
#include "kernel/osl/services_shared.h"

namespace DeviceStrings {

/* "" */
ccl_device_constant DeviceString _emptystring_ = 0ull;
/* "common" */
ccl_device_constant DeviceString u_common = 14645198576927606093ull;
/* "world" */
ccl_device_constant DeviceString u_world = 16436542438370751598ull;
/* "shader" */
ccl_device_constant DeviceString u_shader = 4279676006089868ull;
/* "object" */
ccl_device_constant DeviceString u_object = 973692718279674627ull;
/* "NDC" */
ccl_device_constant DeviceString u_ndc = 5148305047403260775ull;
/* "screen" */
ccl_device_constant DeviceString u_screen = 14159088609039777114ull;
/* "camera" */
ccl_device_constant DeviceString u_camera = 2159505832145726196ull;
/* "raster" */
ccl_device_constant DeviceString u_raster = 7759263238610201778ull;
/* "hsv" */
ccl_device_constant DeviceString u_hsv = 2177035556331879497ull;
/* "hsl" */
ccl_device_constant DeviceString u_hsl = 7749766809258288148ull;
/* "XYZ" */
ccl_device_constant DeviceString u_xyz = 4957977063494975483ull;
/* "xyY" */
ccl_device_constant DeviceString u_xyy = 5138822319725660255ull;
/* "sRGB" */
ccl_device_constant DeviceString u_srgb = 15368599878474175032ull;
/* "object:location" */
ccl_device_constant DeviceString u_object_location = 7846190347358762897ull;
/* "object:color" */
ccl_device_constant DeviceString u_object_color = 12695623857059169556ull;
/* "object:alpha" */
ccl_device_constant DeviceString u_object_alpha = 11165053919428293151ull;
/* "object:index" */
ccl_device_constant DeviceString u_object_index = 6588325838217472556ull;
/* "object:is_light" */
ccl_device_constant DeviceString u_object_is_light = 13979755312845091842ull;
/* "geom:bump_map_normal" */
ccl_device_constant DeviceString u_bump_map_normal = 9592102745179132106ull;
/* "geom:dupli_generated" */
ccl_device_constant DeviceString u_geom_dupli_generated = 6715607178003388908ull;
/* "geom:dupli_uv" */
ccl_device_constant DeviceString u_geom_dupli_uv = 1294253317490155849ull;
/* "material:index" */
ccl_device_constant DeviceString u_material_index = 741770758159634623ull;
/* "object:random" */
ccl_device_constant DeviceString u_object_random = 15789063994977955884ull;
/* "particle:index" */
ccl_device_constant DeviceString u_particle_index = 9489711748229903784ull;
/* "particle:random" */
ccl_device_constant DeviceString u_particle_random = 17993722202766855761ull;
/* "particle:age" */
ccl_device_constant DeviceString u_particle_age = 7380730644710951109ull;
/* "particle:lifetime" */
ccl_device_constant DeviceString u_particle_lifetime = 16576828923156200061ull;
/* "particle:location" */
ccl_device_constant DeviceString u_particle_location = 10309536211423573010ull;
/* "particle:rotation" */
ccl_device_constant DeviceString u_particle_rotation = 17858543768041168459ull;
/* "particle:size" */
ccl_device_constant DeviceString u_particle_size = 16461524249715420389ull;
/* "particle:velocity" */
ccl_device_constant DeviceString u_particle_velocity = 13199101248768308863ull;
/* "particle:angular_velocity" */
ccl_device_constant DeviceString u_particle_angular_velocity = 16327930120486517910ull;
/* "geom:numpolyvertices" */
ccl_device_constant DeviceString u_geom_numpolyvertices = 382043551489988826ull;
/* "geom:trianglevertices" */
ccl_device_constant DeviceString u_geom_trianglevertices = 17839267571524187074ull;
/* "geom:polyvertices" */
ccl_device_constant DeviceString u_geom_polyvertices = 1345577201967881769ull;
/* "geom:name" */
ccl_device_constant DeviceString u_geom_name = 13606338128269760050ull;
/* "geom:undisplaced" */
ccl_device_constant DeviceString u_geom_undisplaced = 12431586303019276305ull;
/* "geom:is_smooth" */
ccl_device_constant DeviceString u_is_smooth = 857544214094480123ull;
/* "geom:is_curve" */
ccl_device_constant DeviceString u_is_curve = 129742495633653138ull;
/* "geom:curve_thickness" */
ccl_device_constant DeviceString u_curve_thickness = 10605802038397633852ull;
/* "geom:curve_length" */
ccl_device_constant DeviceString u_curve_length = 11423459517663715453ull;
/* "geom:curve_tangent_normal" */
ccl_device_constant DeviceString u_curve_tangent_normal = 12301397394034985633ull;
/* "geom:curve_random" */
ccl_device_constant DeviceString u_curve_random = 15293085049960492358ull;
/* "geom:is_point" */
ccl_device_constant DeviceString u_is_point = 2511357849436175953ull;
/* "geom:point_radius" */
ccl_device_constant DeviceString u_point_radius = 9956381140398668479ull;
/* "geom:point_position" */
ccl_device_constant DeviceString u_point_position = 15684484280742966916ull;
/* "geom:point_random" */
ccl_device_constant DeviceString u_point_random = 5632627207092325544ull;
/* "geom:normal_map_normal" */
ccl_device_constant DeviceString u_normal_map_normal = 10718948685686827073;
/* "path:ray_length" */
ccl_device_constant DeviceString u_path_ray_length = 16391985802412544524ull;
/* "path:ray_depth" */
ccl_device_constant DeviceString u_path_ray_depth = 16643933224879500399ull;
/* "path:diffuse_depth" */
ccl_device_constant DeviceString u_path_diffuse_depth = 13191651286699118408ull;
/* "path:glossy_depth" */
ccl_device_constant DeviceString u_path_glossy_depth = 15717768399057252940ull;
/* "path:transparent_depth" */
ccl_device_constant DeviceString u_path_transparent_depth = 7821650266475578543ull;
/* "path:transmission_depth" */
ccl_device_constant DeviceString u_path_transmission_depth = 15113408892323917624ull;
/* "cam:sensor_size" */
ccl_device_constant DeviceString u_sensor_size = 7525693591727141378ull;
/* "cam:image_resolution" */
ccl_device_constant DeviceString u_image_resolution = 5199143367706113607ull;
/* "cam:aperture_aspect_ratio" */
ccl_device_constant DeviceString u_aperture_aspect_ratio = 8708221138893210943ull;
/* "cam:aperture_size" */
ccl_device_constant DeviceString u_aperture_size = 3708482920470008383ull;
/* "cam:aperture_position" */
ccl_device_constant DeviceString u_aperture_position = 12926784411960338650ull;
/* "cam:focal_distance" */
ccl_device_constant DeviceString u_focal_distance = 7162995161881858159ull;

}  // namespace DeviceStrings

/* Closure */

ccl_device_extern ccl_private OSLClosure *osl_mul_closure_color(ccl_private ShaderGlobals *sg,
                                                                ccl_private OSLClosure *a,
                                                                const ccl_private float3 *weight)
{
  if (*weight == zero_float3() || !a) {
    return nullptr;
  }
  else if (*weight == one_float3()) {
    return a;
  }

  ccl_private uint8_t *closure_pool = sg->closure_pool;
  /* Align pointer to closure struct requirement */
  closure_pool = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<size_t>(closure_pool) + alignof(OSLClosureMul) - 1) &
      (-alignof(OSLClosureMul)));
  sg->closure_pool = closure_pool + sizeof(OSLClosureMul);

  ccl_private OSLClosureMul *const closure = reinterpret_cast<ccl_private OSLClosureMul *>(
      closure_pool);
  closure->id = OSL_CLOSURE_MUL_ID;
  closure->weight = *weight;
  closure->closure = a;

  return closure;
}

ccl_device_extern ccl_private OSLClosure *osl_mul_closure_float(ccl_private ShaderGlobals *sg,
                                                                ccl_private OSLClosure *a,
                                                                const float weight)
{
  if (weight == 0.0f || !a) {
    return nullptr;
  }
  else if (weight == 1.0f) {
    return a;
  }

  ccl_private uint8_t *closure_pool = sg->closure_pool;
  /* Align pointer to closure struct requirement */
  closure_pool = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<size_t>(closure_pool) + alignof(OSLClosureMul) - 1) &
      (-alignof(OSLClosureMul)));
  sg->closure_pool = closure_pool + sizeof(OSLClosureMul);

  ccl_private OSLClosureMul *const closure = reinterpret_cast<ccl_private OSLClosureMul *>(
      closure_pool);
  closure->id = OSL_CLOSURE_MUL_ID;
  closure->weight = make_float3(weight, weight, weight);
  closure->closure = a;

  return closure;
}

ccl_device_extern ccl_private OSLClosure *osl_add_closure_closure(ccl_private ShaderGlobals *sg,
                                                                  ccl_private OSLClosure *a,
                                                                  ccl_private OSLClosure *b)
{
  if (!a) {
    return b;
  }
  if (!b) {
    return a;
  }

  ccl_private uint8_t *closure_pool = sg->closure_pool;
  /* Align pointer to closure struct requirement */
  closure_pool = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<size_t>(closure_pool) + alignof(OSLClosureAdd) - 1) &
      (-alignof(OSLClosureAdd)));
  sg->closure_pool = closure_pool + sizeof(OSLClosureAdd);

  ccl_private OSLClosureAdd *const closure = reinterpret_cast<ccl_private OSLClosureAdd *>(
      closure_pool);
  closure->id = OSL_CLOSURE_ADD_ID;
  closure->closureA = a;
  closure->closureB = b;

  return closure;
}

ccl_device_extern ccl_private OSLClosure *osl_allocate_closure_component(
    ccl_private ShaderGlobals *sg, const int id, const int size)
{
  ccl_private uint8_t *closure_pool = sg->closure_pool;
  /* Align pointer to closure struct requirement */
  closure_pool = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<size_t>(closure_pool) + alignof(OSLClosureComponent) - 1) &
      (-alignof(OSLClosureComponent)));
  sg->closure_pool = closure_pool + sizeof(OSLClosureComponent) + size;

  ccl_private OSLClosureComponent *const closure =
      reinterpret_cast<ccl_private OSLClosureComponent *>(closure_pool);
  closure->id = static_cast<OSLClosureType>(id);
  closure->weight = one_float3();

  return closure;
}

ccl_device_extern ccl_private OSLClosure *osl_allocate_weighted_closure_component(
    ccl_private ShaderGlobals *sg, const int id, const int size, const ccl_private float3 *weight)
{
  ccl_private uint8_t *closure_pool = sg->closure_pool;
  /* Align pointer to closure struct requirement */
  closure_pool = reinterpret_cast<uint8_t *>(
      (reinterpret_cast<size_t>(closure_pool) + alignof(OSLClosureComponent) - 1) &
      (-alignof(OSLClosureComponent)));
  sg->closure_pool = closure_pool + sizeof(OSLClosureComponent) + size;

  ccl_private OSLClosureComponent *const closure =
      reinterpret_cast<ccl_private OSLClosureComponent *>(closure_pool);
  closure->id = static_cast<OSLClosureType>(id);
  closure->weight = *weight;

  return closure;
}

/* Utilities */

ccl_device_extern void osl_error(ccl_private ShaderGlobals *sg, const char *format, void *args) {}

ccl_device_extern void osl_printf(ccl_private ShaderGlobals *sg, const char *format, void *args) {}

ccl_device_extern void osl_warning(ccl_private ShaderGlobals *sg, const char *format, void *args)
{
}

ccl_device_extern uint osl_range_check(const int indexvalue,
                                       const int length,
                                       DeviceString symname,
                                       ccl_private ShaderGlobals *sg,
                                       DeviceString sourcefile,
                                       const int sourceline,
                                       DeviceString groupname,
                                       const int layer,
                                       DeviceString layername,
                                       DeviceString shadername)
{
  const int result = indexvalue < 0 ? 0 : indexvalue >= length ? length - 1 : indexvalue;
#if 0
  if (result != indexvalue) {
    printf("Index [%d] out of range\n", indexvalue);
  }
#endif
  return result;
}

ccl_device_extern uint osl_range_check_err(const int indexvalue,
                                           const int length,
                                           DeviceString symname,
                                           ccl_private ShaderGlobals *sg,
                                           DeviceString sourcefile,
                                           const int sourceline,
                                           DeviceString groupname,
                                           const int layer,
                                           DeviceString layername,
                                           DeviceString shadername)
{
  return osl_range_check(indexvalue,
                         length,
                         symname,
                         sg,
                         sourcefile,
                         sourceline,
                         groupname,
                         layer,
                         layername,
                         shadername);
}

/* Color Utilities */

ccl_device_extern void osl_blackbody_vf(ccl_private ShaderGlobals *sg,
                                        ccl_private float3 *result,
                                        const float temperature)
{
  float3 color_rgb = rec709_to_rgb(nullptr, svm_math_blackbody_color_rec709(temperature));
  color_rgb = max(color_rgb, zero_float3());
  *result = color_rgb;
}

ccl_device_extern void osl_wavelength_color_vf(ccl_private ShaderGlobals *sg,
                                               ccl_private float3 *result,
                                               const float lambda_nm)
{
  float3 color = xyz_to_rgb(nullptr, svm_math_wavelength_color_xyz(lambda_nm));
  color *= 1.0f / 2.52f;  // Empirical scale from lg to make all comps <= 1

  /* Clamp to zero if values are smaller */
  *result = max(color, make_float3(0.0f, 0.0f, 0.0f));
}

ccl_device_extern void osl_luminance_fv(ccl_private ShaderGlobals *sg,
                                        ccl_private float *result,
                                        ccl_private float3 *color)
{
  *result = linear_rgb_to_gray(nullptr, *color);
}

ccl_device_extern void osl_luminance_dfdv(ccl_private ShaderGlobals *sg,
                                          ccl_private float *result,
                                          ccl_private float3 *color)
{
  for (int i = 0; i < 3; ++i) {
    osl_luminance_fv(sg, result + i, color + i);
  }
}

ccl_device_extern void osl_prepend_color_from(ccl_private ShaderGlobals *sg,
                                              ccl_private float3 *res,
                                              DeviceString from)
{
  if (from == DeviceStrings::u_hsv) {
    *res = hsv_to_rgb(*res);
  }
  else if (from == DeviceStrings::u_hsl) {
    *res = hsl_to_rgb(*res);
  }
  else if (from == DeviceStrings::u_xyz) {
    *res = xyz_to_rgb(nullptr, *res);
  }
  else if (from == DeviceStrings::u_xyy) {
    *res = xyz_to_rgb(nullptr, xyY_to_xyz(res->x, res->y, res->z));
  }
}

ccl_device_extern bool osl_transformc(ccl_private ShaderGlobals *sg,
                                      ccl_private float3 *c_in,
                                      int c_in_derivs,
                                      ccl_private float3 *c_out,
                                      const int c_out_derivs,
                                      DeviceString from,
                                      DeviceString to)
{
  if (!c_out_derivs) {
    c_in_derivs = false;
  }
  else if (!c_in_derivs) {
    c_out[1] = zero_float3();
    c_out[2] = zero_float3();
  }

  float3 rgb;

  for (int i = 0; i < (c_in_derivs ? 3 : 1); ++i) {
    if (from == DeviceStrings::u_hsv) {
      rgb = hsv_to_rgb(c_in[i]);
    }
    else if (from == DeviceStrings::u_hsl) {
      rgb = hsl_to_rgb(c_in[i]);
    }
    else if (from == DeviceStrings::u_xyz) {
      rgb = xyz_to_rgb(nullptr, c_in[i]);
    }
    else if (from == DeviceStrings::u_xyy) {
      rgb = xyz_to_rgb(nullptr, xyY_to_xyz(c_in[i].x, c_in[i].y, c_in[i].z));
    }
    else if (from == DeviceStrings::u_srgb) {
      rgb = color_srgb_to_linear_v3(c_in[i]);
    }
    else {
      rgb = c_in[i];
    }

    if (to == DeviceStrings::u_hsv) {
      c_out[i] = rgb_to_hsv(rgb);
    }
    else if (to == DeviceStrings::u_hsl) {
      c_out[i] = rgb_to_hsl(rgb);
    }
#if 0
    else if (to == DeviceStrings::u_xyz) {
      c_out[i] = rgb_to_xyz(nullptr, rgb);
    }
    else if (to == DeviceStrings::u_xyy) {
      c_out[i] = xyz_to_xyY(rgb_to_xyz(nullptr, rgb));
    }
#endif
    else if (to == DeviceStrings::u_srgb) {
      c_out[i] = color_linear_to_srgb_v3(rgb);
    }
    else {
      c_out[i] = rgb;
    }
  }

  return true;
}

/* Matrix Utilities */

ccl_device_forceinline void copy_matrix(ccl_private float *res, const Transform &tfm)
{
  res[0] = tfm.x.x;
  res[1] = tfm.y.x;
  res[2] = tfm.z.x;
  res[3] = 0.0f;
  res[4] = tfm.x.y;
  res[5] = tfm.y.y;
  res[6] = tfm.z.y;
  res[7] = 0.0f;
  res[8] = tfm.x.z;
  res[9] = tfm.y.z;
  res[10] = tfm.z.z;
  res[11] = 0.0f;
  res[12] = tfm.x.w;
  res[13] = tfm.y.w;
  res[14] = tfm.z.w;
  res[15] = 1.0f;
}
ccl_device_forceinline void copy_matrix(ccl_private float *res, const ProjectionTransform &tfm)
{
  res[0] = tfm.x.x;
  res[1] = tfm.y.x;
  res[2] = tfm.z.x;
  res[3] = tfm.w.x;
  res[4] = tfm.x.y;
  res[5] = tfm.y.y;
  res[6] = tfm.z.y;
  res[7] = tfm.w.y;
  res[8] = tfm.x.z;
  res[9] = tfm.y.z;
  res[10] = tfm.z.z;
  res[11] = tfm.w.z;
  res[12] = tfm.x.w;
  res[13] = tfm.y.w;
  res[14] = tfm.z.w;
  res[15] = tfm.w.w;
}
ccl_device_forceinline Transform make_transform(const ccl_private float *m)
{
  return make_transform(
      m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14]);
}
ccl_device_forceinline ProjectionTransform make_projection(const ccl_private float *m)
{
  return make_projection(m[0],
                         m[4],
                         m[8],
                         m[12],
                         m[1],
                         m[5],
                         m[9],
                         m[13],
                         m[2],
                         m[6],
                         m[10],
                         m[14],
                         m[3],
                         m[7],
                         m[11],
                         m[15]);
}

ccl_device_extern void osl_mul_mmm(ccl_private float *res,
                                   const ccl_private float *a,
                                   const ccl_private float *b)
{
  const ProjectionTransform tfm_a = make_projection(a);
  const ProjectionTransform tfm_b = make_projection(b);
  copy_matrix(res, tfm_a * tfm_b);
}

ccl_device_extern void osl_mul_mmf(ccl_private float *res,
                                   const ccl_private float *a,
                                   const float b)
{
  for (int i = 0; i < 16; ++i) {
    res[i] = a[i] * b;
  }
}

ccl_device_extern void osl_div_mmm(ccl_private float *res,
                                   const ccl_private float *a,
                                   const ccl_private float *b)
{
  const ProjectionTransform tfm_a = make_projection(a);
  const ProjectionTransform tfm_b = make_projection(b);
  copy_matrix(res, tfm_a * projection_inverse(tfm_b));
}

ccl_device_extern void osl_div_mmf(ccl_private float *res,
                                   const ccl_private float *a,
                                   const float b)
{
  for (int i = 0; i < 16; ++i) {
    res[i] = a[i] / b;
  }
}

ccl_device_extern void osl_div_mfm(ccl_private float *res,
                                   const float a,
                                   const ccl_private float *b)
{
  const ProjectionTransform tfm_b = make_projection(b);
  copy_matrix(res, projection_inverse(tfm_b));
  for (int i = 0; i < 16; ++i) {
    res[i] *= a;
  }
}

ccl_device_extern void osl_div_m_ff(ccl_private float *res, const float a, float b)
{
  float f = (b == 0) ? 0.0f : (a / b);
  copy_matrix(res, projection_identity());
  for (int i = 0; i < 16; ++i) {
    res[i] *= f;
  }
}

ccl_device_extern void osl_transform_vmv(ccl_private float3 *res,
                                         const ccl_private float *m,
                                         const ccl_private float3 *v)
{
  const ProjectionTransform tfm_m = make_projection(m);
  *res = transform_perspective(&tfm_m, *v);
}

ccl_device_extern void osl_transform_dvmdv(ccl_private float3 *res,
                                           const ccl_private float *m,
                                           const ccl_private float3 *v)
{
  const ProjectionTransform tfm_m = make_projection(m);
  res[0] = transform_perspective_deriv(&tfm_m, v[0], v[1], v[2], res[1], res[2]);
}

ccl_device_extern void osl_transformv_vmv(ccl_private float3 *res,
                                          const ccl_private float *m,
                                          const ccl_private float3 *v)
{
  const Transform tfm_m = make_transform(m);
  *res = transform_direction(&tfm_m, *v);
}

ccl_device_extern void osl_transformv_dvmdv(ccl_private float3 *res,
                                            const ccl_private float *m,
                                            const ccl_private float3 *v)
{
  const Transform tfm_m = make_transform(m);
  for (int i = 0; i < 3; ++i) {
    res[i] = transform_direction(&tfm_m, v[i]);
  }
}

ccl_device_extern void osl_transformn_vmv(ccl_private float3 *res,
                                          const ccl_private float *m,
                                          const ccl_private float3 *v)
{
  const Transform tfm_m = transform_inverse(make_transform(m));
  *res = transform_direction_transposed(&tfm_m, *v);
}

ccl_device_extern void osl_transformn_dvmdv(ccl_private float3 *res,
                                            const ccl_private float *m,
                                            const ccl_private float3 *v)
{
  const Transform tfm_m = transform_inverse(make_transform(m));
  for (int i = 0; i < 3; ++i) {
    res[i] = transform_direction_transposed(&tfm_m, v[i]);
  }
}

ccl_device_extern bool osl_get_matrix(ccl_private ShaderGlobals *sg,
                                      ccl_private float *res,
                                      DeviceString from)
{
  if (from == DeviceStrings::u_common || from == DeviceStrings::u_world) {
    copy_matrix(res, projection_identity());
    return true;
  }
  if (from == DeviceStrings::u_shader || from == DeviceStrings::u_object) {
    KernelGlobals kg = nullptr;
    ccl_private ShaderData *const sd = sg->sd;
    int object = sd->object;

    if (object != OBJECT_NONE) {
      const Transform tfm = object_get_transform(kg, sd);
      copy_matrix(res, tfm);
      return true;
    }
  }
  else if (from == DeviceStrings::u_ndc) {
    copy_matrix(res, kernel_data.cam.ndctoworld);
    return true;
  }
  else if (from == DeviceStrings::u_raster) {
    copy_matrix(res, kernel_data.cam.rastertoworld);
    return true;
  }
  else if (from == DeviceStrings::u_screen) {
    copy_matrix(res, kernel_data.cam.screentoworld);
    return true;
  }
  else if (from == DeviceStrings::u_camera) {
    copy_matrix(res, kernel_data.cam.cameratoworld);
    return true;
  }

  return false;
}

ccl_device_extern bool osl_get_inverse_matrix(ccl_private ShaderGlobals *sg,
                                              ccl_private float *res,
                                              DeviceString to)
{
  if (to == DeviceStrings::u_common || to == DeviceStrings::u_world) {
    copy_matrix(res, projection_identity());
    return true;
  }
  if (to == DeviceStrings::u_shader || to == DeviceStrings::u_object) {
    KernelGlobals kg = nullptr;
    ccl_private ShaderData *const sd = sg->sd;
    int object = sd->object;

    if (object != OBJECT_NONE) {
      const Transform itfm = object_get_inverse_transform(kg, sd);
      copy_matrix(res, itfm);
      return true;
    }
  }
  else if (to == DeviceStrings::u_ndc) {
    copy_matrix(res, kernel_data.cam.worldtondc);
    return true;
  }
  else if (to == DeviceStrings::u_raster) {
    copy_matrix(res, kernel_data.cam.worldtoraster);
    return true;
  }
  else if (to == DeviceStrings::u_screen) {
    copy_matrix(res, kernel_data.cam.worldtoscreen);
    return true;
  }
  else if (to == DeviceStrings::u_camera) {
    copy_matrix(res, kernel_data.cam.worldtocamera);
    return true;
  }

  return false;
}

ccl_device_extern bool osl_prepend_matrix_from(ccl_private ShaderGlobals *sg,
                                               ccl_private float *res,
                                               DeviceString from)
{
  float m_from[16];
  if (osl_get_matrix(sg, m_from, from)) {
    osl_mul_mmm(res, m_from, res);
    return true;
  }

  return false;
}

ccl_device_extern bool osl_get_from_to_matrix(ccl_private ShaderGlobals *sg,
                                              ccl_private float *res,
                                              DeviceString from,
                                              DeviceString to)
{
  float m_from[16], m_to[16];
  if (osl_get_matrix(sg, m_from, from) && osl_get_inverse_matrix(sg, m_to, to)) {
    osl_mul_mmm(res, m_from, m_to);
    return true;
  }

  return false;
}

ccl_device_extern bool osl_transform_triple(ccl_private ShaderGlobals *sg,
                                            ccl_private float3 *p_in,
                                            int p_in_derivs,
                                            ccl_private float3 *p_out,
                                            const int p_out_derivs,
                                            DeviceString from,
                                            DeviceString to,
                                            const int vectype)
{
  if (!p_out_derivs) {
    p_in_derivs = false;
  }
  else if (!p_in_derivs) {
    p_out[1] = zero_float3();
    p_out[2] = zero_float3();
  }

  bool res;
  float m[16];

  if (from == DeviceStrings::u_common) {
    res = osl_get_inverse_matrix(sg, m, to);
  }
  else if (to == DeviceStrings::u_common) {
    res = osl_get_matrix(sg, m, from);
  }
  else {
    res = osl_get_from_to_matrix(sg, m, from, to);
  }

  if (res) {
    if (vectype == 2 /* TypeDesc::POINT */) {
      if (p_in_derivs) {
        osl_transform_dvmdv(p_out, m, p_in);
      }
      else {
        osl_transform_vmv(p_out, m, p_in);
      }
    }
    else if (vectype == 3 /* TypeDesc::VECTOR */) {
      if (p_in_derivs) {
        osl_transformv_dvmdv(p_out, m, p_in);
      }
      else {
        osl_transformv_vmv(p_out, m, p_in);
      }
    }
    else if (vectype == 4 /* TypeDesc::NORMAL */) {
      if (p_in_derivs) {
        osl_transformn_dvmdv(p_out, m, p_in);
      }
      else {
        osl_transformn_vmv(p_out, m, p_in);
      }
    }
    else {
      res = false;
    }
  }
  else {
    p_out[0] = p_in[0];
    if (p_in_derivs) {
      p_out[1] = p_in[1];
      p_out[2] = p_in[2];
    }
  }

  return res;
}

ccl_device_extern bool osl_transform_triple_nonlinear(ccl_private ShaderGlobals *sg,
                                                      ccl_private float3 *p_in,
                                                      const int p_in_derivs,
                                                      ccl_private float3 *p_out,
                                                      const int p_out_derivs,
                                                      DeviceString from,
                                                      DeviceString to,
                                                      const int vectype)
{
  return osl_transform_triple(sg, p_in, p_in_derivs, p_out, p_out_derivs, from, to, vectype);
}

ccl_device_extern void osl_transpose_mm(ccl_private float *res, const ccl_private float *m)
{
  copy_matrix(res, *reinterpret_cast<const ccl_private ProjectionTransform *>(m));
}

#if 0
ccl_device_extern float osl_determinant_fm(ccl_private const float *m)
{
}
#endif

/* Attributes */

typedef long long TypeDesc;

template<typename T>
ccl_device_inline bool set_attribute(const T v,
                                     const T dx,
                                     const T dy,
                                     const TypeDesc type,
                                     bool derivatives,
                                     ccl_private void *val);

ccl_device_inline void set_data_float(
    const float v, const float dx, const float dy, bool derivatives, ccl_private void *val)
{
  ccl_private float *fval = static_cast<ccl_private float *>(val);
  fval[0] = v;
  if (derivatives) {
    fval[1] = dx;
    fval[2] = dy;
  }
}

ccl_device_inline void set_data_float3(
    const float3 v, const float3 dx, const float3 dy, bool derivatives, ccl_private void *val)
{
  ccl_private float *fval = static_cast<ccl_private float *>(val);
  fval[0] = v.x;
  fval[1] = v.y;
  fval[2] = v.z;
  if (derivatives) {
    fval[3] = dx.x;
    fval[4] = dx.y;
    fval[5] = dx.z;
    fval[6] = dy.x;
    fval[7] = dy.y;
    fval[8] = dy.z;
  }
}

ccl_device_inline void set_data_float4(
    const float4 v, const float4 dx, const float4 dy, bool derivatives, ccl_private void *val)
{
  ccl_private float *fval = static_cast<ccl_private float *>(val);
  fval[0] = v.x;
  fval[1] = v.y;
  fval[2] = v.z;
  fval[3] = v.w;
  if (derivatives) {
    fval[4] = dx.x;
    fval[5] = dx.y;
    fval[6] = dx.z;
    fval[7] = dx.w;
    fval[8] = dy.x;
    fval[9] = dy.y;
    fval[10] = dy.z;
    fval[11] = dy.w;
  }
}

ccl_device_template_spec bool set_attribute(const float v,
                                            const float dx,
                                            const float dy,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;
  const int type_arraylen = type >> 32;

  if (type_basetype == 11 /* TypeDesc::FLOAT */) {
    if ((type_aggregate == 3 /* TypeDesc::VEC3 */) || (type_aggregate == 1 && type_arraylen == 3))
    {
      set_data_float3(make_float3(v), make_float3(dx), make_float3(dy), derivatives, val);
      return true;
    }
    if ((type_aggregate == 4 /* TypeDesc::VEC4 */) || (type_aggregate == 1 && type_arraylen == 4))
    {
      set_data_float4(make_float4(v, v, v, 1.0f),
                      make_float4(dx, dx, dx, 0.0f),
                      make_float4(dy, dy, dy, 0.0f),
                      derivatives,
                      val);
      return true;
    }
    if ((type_aggregate == 1 /* TypeDesc::SCALAR */)) {
      set_data_float(v, dx, dy, derivatives, val);
      return true;
    }
  }

  return false;
}
ccl_device_template_spec bool set_attribute(const float2 v,
                                            const float2 dx,
                                            const float2 dy,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;
  const int type_arraylen = type >> 32;

  if (type_basetype == 11 /* TypeDesc::FLOAT */) {
    if ((type_aggregate == 3 /* TypeDesc::VEC3 */) || (type_aggregate == 1 && type_arraylen == 3))
    {
      set_data_float3(make_float3(v), make_float3(dx), make_float3(dy), derivatives, val);
      return true;
    }
    if ((type_aggregate == 4 /* TypeDesc::VEC4 */) || (type_aggregate == 1 && type_arraylen == 4))
    {
      set_data_float4(make_float4(v.x, v.y, 0.0f, 1.0f),
                      make_float4(dx.x, dx.y, 0.0f, 0.0f),
                      make_float4(dy.x, dy.y, 0.0f, 0.0f),
                      derivatives,
                      val);
    }
    if ((type_aggregate == 1 /* TypeDesc::SCALAR */)) {
      set_data_float(average(v), average(dx), average(dy), derivatives, val);
      return true;
    }
  }

  return false;
}
ccl_device_template_spec bool set_attribute(const float3 v,
                                            const float3 dx,
                                            const float3 dy,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;
  const int type_arraylen = type >> 32;

  if (type_basetype == 11 /* TypeDesc::FLOAT */) {
    if ((type_aggregate == 3 /* TypeDesc::VEC3 */) || (type_aggregate == 1 && type_arraylen == 3))
    {
      set_data_float3(v, dx, dy, derivatives, val);
      return true;
    }
    if ((type_aggregate == 4 /* TypeDesc::VEC4 */) || (type_aggregate == 1 && type_arraylen == 4))
    {
      set_data_float4(
          make_float4(v, 1.0f), make_float4(dx, 0.0f), make_float4(dy, 0.0f), derivatives, val);
      return true;
    }
    if ((type_aggregate == 1 /* TypeDesc::SCALAR */)) {
      set_data_float(average(v), average(dx), average(dy), derivatives, val);
      return true;
    }
  }

  return false;
}
ccl_device_template_spec bool set_attribute(const float4 v,
                                            const float4 dx,
                                            const float4 dy,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;
  const int type_arraylen = type >> 32;

  if (type_basetype == 11 /* TypeDesc::FLOAT */) {
    if ((type_aggregate == 3 /* TypeDesc::VEC3 */) || (type_aggregate == 1 && type_arraylen == 3))
    {
      set_data_float3(make_float3(v), make_float3(dx), make_float3(dy), derivatives, val);
      return true;
    }
    if ((type_aggregate == 4 /* TypeDesc::VEC4 */) || (type_aggregate == 1 && type_arraylen == 4))
    {
      set_data_float4(v, dx, dy, derivatives, val);
      return true;
    }
    if ((type_aggregate == 1 /* TypeDesc::SCALAR */)) {
      set_data_float(average(make_float3(v)),
                     average(make_float3(dx)),
                     average(make_float3(dy)),
                     derivatives,
                     val);
      return true;
    }
  }

  return false;
}

template<typename T>
ccl_device_inline bool set_attribute(const T f,
                                     const TypeDesc type,
                                     bool derivatives,
                                     ccl_private void *val)
{
  return set_attribute(f, make_zero<T>(), make_zero<T>(), type, derivatives, val);
}

ccl_device_inline bool set_attribute_matrix(const ccl_private Transform &tfm,
                                            const TypeDesc type,
                                            ccl_private void *val)
{
  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;

  if (type_basetype == 11 /* TypeDesc::FLOAT */ && type_aggregate == 16 /* TypeDesc::MATRIX44 */) {
    copy_matrix(static_cast<ccl_private float *>(val), tfm);
    return true;
  }

  return false;
}

ccl_device_template_spec bool set_attribute(const int i,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  ccl_private int *ival = static_cast<ccl_private int *>(val);

  const unsigned char type_basetype = type & 0xFF;
  const unsigned char type_aggregate = (type >> 8) & 0xFF;
  const int type_arraylen = type >> 32;

  if ((type_basetype == 7 /* TypeDesc::INT */) && (type_aggregate == 1 /* TypeDesc::SCALAR */) &&
      type_arraylen == 0)
  {
    ival[0] = i;

    if (derivatives) {
      ival[1] = 0;
      ival[2] = 0;
    }

    return true;
  }

  return false;
}

ccl_device_inline bool get_background_attribute(KernelGlobals kg,
                                                ccl_private ShaderGlobals *sg,
                                                ccl_private ShaderData *sd,
                                                DeviceString name,
                                                const TypeDesc type,
                                                bool derivatives,
                                                ccl_private void *val)
{
  ConstIntegratorState state = (sg->shade_index > 0) ? (sg->shade_index - 1) : -1;
  ConstIntegratorShadowState shadow_state = (sg->shade_index < 0) ? (-sg->shade_index - 1) : -1;
  if (name == DeviceStrings::u_path_ray_length) {
    /* Ray Length */
    float f = sd->ray_length;
    return set_attribute(f, type, derivatives, val);
  }

#define READ_PATH_STATE(elem) \
  ((state != -1)        ? INTEGRATOR_STATE(state, path, elem) : \
   (shadow_state != -1) ? INTEGRATOR_STATE(shadow_state, shadow_path, elem) : \
                          0)

  if (name == DeviceStrings::u_path_ray_depth) {
    /* Ray Depth */
    int f = READ_PATH_STATE(bounce);

    /* Read bounce from different locations depending on if this is a shadow path. For background,
     * light emission and shadow evaluation from a surface or volume we are effectively one bounce
     * further. */
    if (sg->raytype & (PATH_RAY_SHADOW | PATH_RAY_EMISSION)) {
      f += 1;
    }

    return set_attribute(f, type, derivatives, val);
  }
  if (name == DeviceStrings::u_path_diffuse_depth) {
    /* Diffuse Ray Depth */
    const int f = READ_PATH_STATE(diffuse_bounce);
    return set_attribute(f, type, derivatives, val);
  }
  if (name == DeviceStrings::u_path_glossy_depth) {
    /* Glossy Ray Depth */
    const int f = READ_PATH_STATE(glossy_bounce);
    return set_attribute(f, type, derivatives, val);
  }
  if (name == DeviceStrings::u_path_transmission_depth) {
    /* Transmission Ray Depth */
    const int f = READ_PATH_STATE(transmission_bounce);
    return set_attribute(f, type, derivatives, val);
  }
  if (name == DeviceStrings::u_path_transparent_depth) {
    /* Transparent Ray Depth */
    const int f = READ_PATH_STATE(transparent_bounce);
    return set_attribute(f, type, derivatives, val);
  }
#undef READ_PATH_STATE

  else if (name == DeviceStrings::u_ndc) {
    /* NDC coordinates with special exception for orthographic projection. */
    float3 ndc[3];

    if ((sg->raytype & PATH_RAY_CAMERA) && sd->object == OBJECT_NONE &&
        kernel_data.cam.type == CAMERA_ORTHOGRAPHIC)
    {
      ndc[0] = camera_world_to_ndc(kg, sd, sd->ray_P);

      if (derivatives) {
        ndc[1] = zero_float3();
        ndc[2] = zero_float3();
      }
    }
    else {
      ndc[0] = camera_world_to_ndc(kg, sd, sd->P);

      if (derivatives) {
        const differential3 dP = differential_from_compact(sd->Ng, sd->dP);
        ndc[1] = camera_world_to_ndc(kg, sd, sd->P + dP.dx) - ndc[0];
        ndc[2] = camera_world_to_ndc(kg, sd, sd->P + dP.dy) - ndc[0];
      }
    }

    return set_attribute(ndc[0], ndc[1], ndc[2], type, derivatives, val);
  }

  return false;
}

template<typename T>
ccl_device_inline bool get_object_attribute_impl(KernelGlobals kg,
                                                 ccl_private ShaderData *sd,
                                                 const AttributeDescriptor &desc,
                                                 const TypeDesc type,
                                                 bool derivatives,
                                                 ccl_private void *val)
{
  T v;
  T dx = make_zero<T>();
  T dy = make_zero<T>();
#ifdef __VOLUME__
  if (primitive_is_volume_attribute(sd)) {
    v = primitive_volume_attribute<T>(kg, sd, desc);
  }
  else
#endif
  {
    v = primitive_surface_attribute<T>(
        kg, sd, desc, derivatives ? &dx : nullptr, derivatives ? &dy : nullptr);
  }
  return set_attribute(v, dx, dy, type, derivatives, val);
}

ccl_device_inline bool get_object_attribute(KernelGlobals kg,
                                            ccl_private ShaderData *sd,
                                            const AttributeDescriptor &desc,
                                            const TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  if (desc.type == NODE_ATTR_FLOAT) {
    return get_object_attribute_impl<float>(kg, sd, desc, type, derivatives, val);
  }
  else if (desc.type == NODE_ATTR_FLOAT2) {
    return get_object_attribute_impl<float2>(kg, sd, desc, type, derivatives, val);
  }
  else if (desc.type == NODE_ATTR_FLOAT3) {
    return get_object_attribute_impl<float3>(kg, sd, desc, type, derivatives, val);
  }
  else if (desc.type == NODE_ATTR_FLOAT4 || desc.type == NODE_ATTR_RGBA) {
    return get_object_attribute_impl<float4>(kg, sd, desc, type, derivatives, val);
  }
  else if (desc.type == NODE_ATTR_MATRIX) {
    Transform tfm = primitive_attribute_matrix(kg, desc);
    return set_attribute_matrix(tfm, type, val);
  }

  return false;
}

ccl_device_inline bool get_object_standard_attribute(KernelGlobals kg,
                                                     ccl_private ShaderGlobals *sg,
                                                     ccl_private ShaderData *sd,
                                                     DeviceString name,
                                                     const TypeDesc type,
                                                     bool derivatives,
                                                     ccl_private void *val)
{
  /* Object attributes */
  if (name == DeviceStrings::u_object_location) {
    float3 f = object_location(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_object_color) {
    float3 f = object_color(kg, sd->object);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_object_alpha) {
    float f = object_alpha(kg, sd->object);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_object_index) {
    float f = object_pass_id(kg, sd->object);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_object_is_light) {
    float f = ((sd->type & PRIMITIVE_LAMP) != 0);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_geom_dupli_generated) {
    float3 f = object_dupli_generated(kg, sd->object);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_geom_dupli_uv) {
    float3 f = object_dupli_uv(kg, sd->object);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_material_index) {
    float f = shader_pass_id(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_object_random) {
    const float f = object_random_number(kg, sd->object);

    return set_attribute(f, type, derivatives, val);
  }

  /* Particle attributes */
  else if (name == DeviceStrings::u_particle_index) {
    int particle_id = object_particle_id(kg, sd->object);
    float f = particle_index(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_particle_random) {
    int particle_id = object_particle_id(kg, sd->object);
    float f = hash_uint2_to_float(particle_index(kg, particle_id), 0);
    return set_attribute(f, type, derivatives, val);
  }

  else if (name == DeviceStrings::u_particle_age) {
    int particle_id = object_particle_id(kg, sd->object);
    float f = particle_age(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_particle_lifetime) {
    int particle_id = object_particle_id(kg, sd->object);
    float f = particle_lifetime(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_particle_location) {
    int particle_id = object_particle_id(kg, sd->object);
    float3 f = particle_location(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
#if 0 /* unsupported */
  else if (name == DeviceStrings::u_particle_rotation) {
    int particle_id = object_particle_id(kg, sd->object);
    float4 f = particle_rotation(kg, particle_id);
    return set_attribute4(f, type, derivatives, val);
  }
#endif
  else if (name == DeviceStrings::u_particle_size) {
    int particle_id = object_particle_id(kg, sd->object);
    float f = particle_size(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_particle_velocity) {
    int particle_id = object_particle_id(kg, sd->object);
    float3 f = particle_velocity(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_particle_angular_velocity) {
    int particle_id = object_particle_id(kg, sd->object);
    float3 f = particle_angular_velocity(kg, particle_id);
    return set_attribute(f, type, derivatives, val);
  }

  /* Geometry attributes */
#if 0 /* TODO */
  else if (name == DeviceStrings::u_geom_numpolyvertices) {
    return false;
  }
  else if (name == DeviceStrings::u_geom_trianglevertices ||
            name == DeviceStrings::u_geom_polyvertices) {
    return false;
  }
  else if (name == DeviceStrings::u_geom_name) {
    return false;
  }
#endif
  else if (name == DeviceStrings::u_is_smooth) {
    float f = ((sd->shader & SHADER_SMOOTH_NORMAL) != 0);
    return set_attribute(f, type, derivatives, val);
  }

#ifdef __HAIR__
  /* Hair attributes */
  else if (name == DeviceStrings::u_is_curve) {
    float f = (sd->type & PRIMITIVE_CURVE) != 0;
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_curve_thickness) {
    float f = curve_thickness(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_curve_tangent_normal) {
    float3 f = curve_tangent_normal(sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_curve_random) {
    float f = curve_random(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
#endif

#ifdef __POINTCLOUD__
  /* Point attributes */
  else if (name == DeviceStrings::u_is_point) {
    float f = (sd->type & PRIMITIVE_POINT) != 0;
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_point_radius) {
    float f = point_radius(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_point_position) {
    float3 f = point_position(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_point_random) {
    float f = point_random(kg, sd);
    return set_attribute(f, type, derivatives, val);
  }
#endif

  else if (name == DeviceStrings::u_normal_map_normal) {
    if (sd->type & PRIMITIVE_TRIANGLE) {
      float3 f = triangle_smooth_normal_unnormalized(kg, sd, sd->Ng, sd->prim, sd->u, sd->v);
      return set_attribute(f, type, derivatives, val);
    }
    else {
      return false;
    }
  }
  if (name == DeviceStrings::u_bump_map_normal) {
    float3 f[3];
    if (!attribute_bump_map_normal(kg, sd, f)) {
      return false;
    }
    return set_attribute(f[0], f[1], f[2], type, derivatives, val);
  }

  return get_background_attribute(kg, sg, sd, name, type, derivatives, val);
}

ccl_device_inline bool get_camera_attribute(ccl_private ShaderGlobals *sg,
                                            KernelGlobals kg,
                                            DeviceString name,
                                            TypeDesc type,
                                            bool derivatives,
                                            ccl_private void *val)
{
  if (name == DeviceStrings::u_sensor_size) {
    const float2 sensor = make_float2(kernel_data.cam.sensorwidth, kernel_data.cam.sensorheight);
    return set_attribute(sensor, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_image_resolution) {
    const float2 image = make_float2(kernel_data.cam.width, kernel_data.cam.height);
    return set_attribute(image, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_aperture_aspect_ratio) {
    return set_attribute(1.0f / kernel_data.cam.inv_aperture_ratio, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_aperture_size) {
    return set_attribute(kernel_data.cam.aperturesize, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_aperture_position) {
    /* The random numbers for aperture sampling are packed into N. */
    const float2 rand_lens = make_float2(sg->N.x, sg->N.y);
    const float2 pos = camera_sample_aperture(&kernel_data.cam, rand_lens);
    return set_attribute(pos * kernel_data.cam.aperturesize, type, derivatives, val);
  }
  else if (name == DeviceStrings::u_focal_distance) {
    return set_attribute(kernel_data.cam.focaldistance, type, derivatives, val);
  }
  return false;
}

ccl_device_extern bool osl_get_attribute(ccl_private ShaderGlobals *sg,
                                         const int derivatives,
                                         DeviceString object_name,
                                         DeviceString name,
                                         const int array_lookup,
                                         const int index,
                                         const TypeDesc type,
                                         ccl_private void *res)
{
  KernelGlobals kg = nullptr;
  ccl_private ShaderData *const sd = sg->sd;
  int object;

  if (sd == nullptr) {
    /* Camera shader. */
    return get_camera_attribute(sg, kg, name, type, derivatives, res);
  }

  if (object_name != DeviceStrings::_emptystring_) {
    /* TODO: Get object index from name */
    return false;
  }
  else {
    object = sd->object;
  }

  const AttributeDescriptor desc = find_attribute(kg, object, sd->prim, name);
  if (desc.offset != ATTR_STD_NOT_FOUND) {
    return get_object_attribute(kg, sd, desc, type, derivatives, res);
  }
  else {
    return get_object_standard_attribute(kg, sg, sd, name, type, derivatives, res);
  }
}

#if 0
ccl_device_extern bool osl_bind_interpolated_param(ccl_private ShaderGlobals *sg,
                                                       DeviceString name,
                                                       long long type,
                                                     const int userdata_has_derivs,
                                                       ccl_private void *userdata_data,
                                                     const int symbol_has_derivs,
                                                       ccl_private void *symbol_data,
                                                     const int symbol_data_size,
                                                       ccl_private void *userdata_initialized,
                                                     const int userdata_index)
{
  return false;
}
#endif

/* Noise */

ccl_device_extern uint osl_hash_ii(const int x)
{
  return hash_uint(x);
}

ccl_device_extern uint osl_hash_if(const float x)
{
  return hash_uint(__float_as_uint(x));
}

ccl_device_extern uint osl_hash_iff(const float x, const float y)
{
  return hash_uint2(__float_as_uint(x), __float_as_uint(y));
}

ccl_device_extern uint osl_hash_iv(const ccl_private float3 *v)
{
  return hash_uint3(__float_as_uint(v->x), __float_as_uint(v->y), __float_as_uint(v->z));
}

ccl_device_extern uint osl_hash_ivf(const ccl_private float3 *v, const float w)
{
  return hash_uint4(
      __float_as_uint(v->x), __float_as_uint(v->y), __float_as_uint(v->z), __float_as_uint(w));
}

ccl_device_extern OSLNoiseOptions *osl_get_noise_options(ccl_private ShaderGlobals *sg)
{
  return nullptr;
}

ccl_device_extern void osl_noiseparams_set_anisotropic(ccl_private OSLNoiseOptions *opt,
                                                       const int anisotropic)
{
}

ccl_device_extern void osl_noiseparams_set_do_filter(ccl_private OSLNoiseOptions *opt,
                                                     const int do_filter)
{
}

ccl_device_extern void osl_noiseparams_set_direction(ccl_private OSLNoiseOptions *opt,
                                                     float3 *direction)
{
}

ccl_device_extern void osl_noiseparams_set_bandwidth(ccl_private OSLNoiseOptions *opt,
                                                     const float bandwidth)
{
}

ccl_device_extern void osl_noiseparams_set_impulses(ccl_private OSLNoiseOptions *opt,
                                                    const float impulses)
{
}

#define OSL_NOISE_IMPL(name, op) \
  ccl_device_extern float name##_ff(const float x) \
  { \
    return op##_1d(x); \
  } \
  ccl_device_extern float name##_fff(const float x, const float y) \
  { \
    return op##_2d(make_float2(x, y)); \
  } \
  ccl_device_extern float name##_fv(ccl_private const float3 *v) \
  { \
    return op##_3d(*v); \
  } \
  ccl_device_extern float name##_fvf(ccl_private const float3 *v, const float w) \
  { \
    return op##_4d(make_float4(v->x, v->y, v->z, w)); \
  } \
  ccl_device_extern void name##_vf(ccl_private float3 *res, const float x) \
  { \
    /* TODO: This is not correct. Really need to change the hash function inside the noise \
     * function to spit out a vector instead of a scalar. */ \
    const float n = name##_ff(x); \
    res->x = n; \
    res->y = n; \
    res->z = n; \
  } \
  ccl_device_extern void name##_vff(ccl_private float3 *res, const float x, float y) \
  { \
    const float n = name##_fff(x, y); \
    res->x = n; \
    res->y = n; \
    res->z = n; \
  } \
  ccl_device_extern void name##_vv(ccl_private float3 *res, ccl_private const float3 *v) \
  { \
    const float n = name##_fv(v); \
    res->x = n; \
    res->y = n; \
    res->z = n; \
  } \
  ccl_device_extern void name##_vvf( \
      ccl_private float3 *res, ccl_private const float3 *v, const float w) \
  { \
    const float n = name##_fvf(v, w); \
    res->x = n; \
    res->y = n; \
    res->z = n; \
  } \
  ccl_device_extern void name##_dfdf(ccl_private float *res, ccl_private const float *x) \
  { \
    res[0] = name##_ff(x[0]); \
    res[1] = name##_ff(x[1]); \
    res[2] = name##_ff(x[2]); \
  } \
  ccl_device_extern void name##_dfdff( \
      ccl_private float *res, ccl_private const float *x, const float y) \
  { \
    res[0] = name##_fff(x[0], y); \
    res[1] = name##_fff(x[1], y); \
    res[2] = name##_fff(x[2], y); \
  } \
  ccl_device_extern void name##_dffdf( \
      ccl_private float *res, const float x, ccl_private const float *y) \
  { \
    res[0] = name##_fff(x, y[0]); \
    res[1] = name##_fff(x, y[1]); \
    res[2] = name##_fff(x, y[2]); \
  } \
  ccl_device_extern void name##_dfdfdf( \
      ccl_private float *res, ccl_private const float *x, ccl_private const float *y) \
  { \
    res[0] = name##_fff(x[0], y[0]); \
    res[1] = name##_fff(x[1], y[1]); \
    res[2] = name##_fff(x[2], y[2]); \
  } \
  ccl_device_extern void name##_dfdv(ccl_private float *res, ccl_private const float3 *v) \
  { \
    res[0] = name##_fv(&v[0]); \
    res[1] = name##_fv(&v[1]); \
    res[2] = name##_fv(&v[2]); \
  } \
  ccl_device_extern void name##_dfdvf( \
      ccl_private float *res, ccl_private const float3 *v, const float w) \
  { \
    res[0] = name##_fvf(&v[0], w); \
    res[1] = name##_fvf(&v[1], w); \
    res[2] = name##_fvf(&v[2], w); \
  } \
  ccl_device_extern void name##_dfvdf( \
      ccl_private float *res, ccl_private const float3 *v, ccl_private const float *w) \
  { \
    res[0] = name##_fvf(v, w[0]); \
    res[1] = name##_fvf(v, w[1]); \
    res[2] = name##_fvf(v, w[2]); \
  } \
  ccl_device_extern void name##_dfdvdf( \
      ccl_private float *res, ccl_private const float3 *v, ccl_private const float *w) \
  { \
    res[0] = name##_fvf(&v[0], w[0]); \
    res[1] = name##_fvf(&v[1], w[1]); \
    res[2] = name##_fvf(&v[2], w[2]); \
  } \
  ccl_device_extern void name##_dvdf(ccl_private float3 *res, ccl_private const float *x) \
  { \
    name##_vf(&res[0], x[0]); \
    name##_vf(&res[1], x[1]); \
    name##_vf(&res[2], x[2]); \
  } \
  ccl_device_extern void name##_dvdff( \
      ccl_private float3 *res, ccl_private const float *x, const float y) \
  { \
    name##_vff(&res[0], x[0], y); \
    name##_vff(&res[1], x[1], y); \
    name##_vff(&res[2], x[2], y); \
  } \
  ccl_device_extern void name##_dvfdf( \
      ccl_private float3 *res, const float x, ccl_private const float *y) \
  { \
    name##_vff(&res[0], x, y[0]); \
    name##_vff(&res[1], x, y[1]); \
    name##_vff(&res[2], x, y[2]); \
  } \
  ccl_device_extern void name##_dvdfdf( \
      ccl_private float3 *res, ccl_private const float *x, ccl_private const float *y) \
  { \
    name##_vff(&res[0], x[0], y[0]); \
    name##_vff(&res[1], x[1], y[1]); \
    name##_vff(&res[2], x[2], y[2]); \
  } \
  ccl_device_extern void name##_dvdv(ccl_private float3 *res, ccl_private const float3 *v) \
  { \
    name##_vv(&res[0], &v[0]); \
    name##_vv(&res[1], &v[1]); \
    name##_vv(&res[2], &v[2]); \
  } \
  ccl_device_extern void name##_dvdvf( \
      ccl_private float3 *res, ccl_private const float3 *v, const float w) \
  { \
    name##_vvf(&res[0], &v[0], w); \
    name##_vvf(&res[1], &v[1], w); \
    name##_vvf(&res[2], &v[2], w); \
  } \
  ccl_device_extern void name##_dvvdf( \
      ccl_private float3 *res, ccl_private const float3 *v, ccl_private const float *w) \
  { \
    name##_vvf(&res[0], v, w[0]); \
    name##_vvf(&res[1], v, w[1]); \
    name##_vvf(&res[2], v, w[2]); \
  } \
  ccl_device_extern void name##_dvdvdf( \
      ccl_private float3 *res, ccl_private const float3 *v, ccl_private const float *w) \
  { \
    name##_vvf(&res[0], &v[0], w[0]); \
    name##_vvf(&res[1], &v[1], w[1]); \
    name##_vvf(&res[2], &v[2], w[2]); \
  }

ccl_device_forceinline float hashnoise_1d(const float p)
{
  const uint x = __float_as_uint(p);
  return hash_uint(x) / static_cast<float>(~0u);
}
ccl_device_forceinline float hashnoise_2d(const float2 p)
{
  const uint x = __float_as_uint(p.x);
  const uint y = __float_as_uint(p.y);
  return hash_uint2(x, y) / static_cast<float>(~0u);
}
ccl_device_forceinline float hashnoise_3d(const float3 p)
{
  const uint x = __float_as_uint(p.x);
  const uint y = __float_as_uint(p.y);
  const uint z = __float_as_uint(p.z);
  return hash_uint3(x, y, z) / static_cast<float>(~0u);
}
ccl_device_forceinline float hashnoise_4d(const float4 p)
{
  const uint x = __float_as_uint(p.x);
  const uint y = __float_as_uint(p.y);
  const uint z = __float_as_uint(p.z);
  const uint w = __float_as_uint(p.w);
  return hash_uint4(x, y, z, w) / static_cast<float>(~0u);
}

/* TODO: Implement all noise functions */
OSL_NOISE_IMPL(osl_hashnoise, hashnoise)
OSL_NOISE_IMPL(osl_noise, noise)
OSL_NOISE_IMPL(osl_snoise, snoise)

/* Texturing */

ccl_device_extern void osl_init_texture_options(ccl_private ShaderGlobals *sg, void *opt) {}

ccl_device_extern ccl_private OSLTextureOptions *osl_get_texture_options(
    ccl_private ShaderGlobals *sg)
{
  return nullptr;
}

ccl_device_extern void osl_texture_set_firstchannel(ccl_private OSLTextureOptions *opt,
                                                    const int firstchannel)
{
}

ccl_device_extern void osl_texture_set_swrap_code(ccl_private OSLTextureOptions *opt,
                                                  const int mode)
{
}

ccl_device_extern void osl_texture_set_twrap_code(ccl_private OSLTextureOptions *opt,
                                                  const int mode)
{
}

ccl_device_extern void osl_texture_set_rwrap_code(ccl_private OSLTextureOptions *opt,
                                                  const int mode)
{
}

ccl_device_extern void osl_texture_set_stwrap_code(ccl_private OSLTextureOptions *opt,
                                                   const int mode)
{
}

ccl_device_extern void osl_texture_set_sblur(ccl_private OSLTextureOptions *opt, const float blur)
{
}

ccl_device_extern void osl_texture_set_tblur(ccl_private OSLTextureOptions *opt, const float blur)
{
}

ccl_device_extern void osl_texture_set_rblur(ccl_private OSLTextureOptions *opt, const float blur)
{
}

ccl_device_extern void osl_texture_set_stblur(ccl_private OSLTextureOptions *opt, const float blur)
{
}

ccl_device_extern void osl_texture_set_swidth(ccl_private OSLTextureOptions *opt,
                                              const float width)
{
}

ccl_device_extern void osl_texture_set_twidth(ccl_private OSLTextureOptions *opt,
                                              const float width)
{
}

ccl_device_extern void osl_texture_set_rwidth(ccl_private OSLTextureOptions *opt,
                                              const float width)
{
}

ccl_device_extern void osl_texture_set_stwidth(ccl_private OSLTextureOptions *opt,
                                               const float width)
{
}

ccl_device_extern void osl_texture_set_fill(ccl_private OSLTextureOptions *opt, const float fill)
{
}

ccl_device_extern void osl_texture_set_time(ccl_private OSLTextureOptions *opt, const float time)
{
}

ccl_device_extern void osl_texture_set_interp_code(ccl_private OSLTextureOptions *opt,
                                                   const int mode)
{
}

ccl_device_extern void osl_texture_set_subimage(ccl_private OSLTextureOptions *opt,
                                                const int subimage)
{
}

ccl_device_extern void osl_texture_set_missingcolor_arena(ccl_private OSLTextureOptions *opt,
                                                          ccl_private float3 *color)
{
}

ccl_device_extern void osl_texture_set_missingcolor_alpha(ccl_private OSLTextureOptions *opt,
                                                          const int nchannels,
                                                          const float alpha)
{
}

ccl_device_extern bool osl_texture(ccl_private ShaderGlobals *sg,
                                   DeviceString filename,
                                   ccl_private void *texture_handle,
                                   ccl_private OSLTextureOptions *opt,
                                   const float s,
                                   const float t,
                                   const float dsdx,
                                   const float dtdx,
                                   const float dsdy,
                                   const float dtdy,
                                   const int nchannels,
                                   ccl_private float *result,
                                   ccl_private float *dresultdx,
                                   ccl_private float *dresultdy,
                                   ccl_private float *alpha,
                                   ccl_private float *dalphadx,
                                   ccl_private float *dalphady,
                                   ccl_private void *errormessage)
{
  const unsigned int type = OSL_TEXTURE_HANDLE_TYPE(texture_handle);
  const unsigned int slot = OSL_TEXTURE_HANDLE_SLOT(texture_handle);

  switch (type) {
    case OSL_TEXTURE_HANDLE_TYPE_SVM: {
      const float4 rgba = kernel_tex_image_interp(nullptr, slot, s, 1.0f - t);
      if (nchannels > 0) {
        result[0] = rgba.x;
      }
      if (nchannels > 1) {
        result[1] = rgba.y;
      }
      if (nchannels > 2) {
        result[2] = rgba.z;
      }
      if (alpha) {
        *alpha = rgba.w;
      }
      return true;
    }
    case OSL_TEXTURE_HANDLE_TYPE_IES: {
      if (nchannels > 0) {
        result[0] = kernel_ies_interp(nullptr, slot, s, t);
      }
      return true;
    }
    default: {
      return false;
    }
  }
}

ccl_device_extern bool osl_texture3d(ccl_private ShaderGlobals *sg,
                                     DeviceString filename,
                                     ccl_private void *texture_handle,
                                     ccl_private OSLTextureOptions *opt,
                                     const ccl_private float3 *P,
                                     const ccl_private float3 *dPdx,
                                     const ccl_private float3 *dPdy,
                                     const ccl_private float3 *dPdz,
                                     const int nchannels,
                                     ccl_private float *result,
                                     ccl_private float *dresultds,
                                     ccl_private float *dresultdt,
                                     ccl_private float *alpha,
                                     ccl_private float *dalphadx,
                                     ccl_private float *dalphady,
                                     ccl_private void *errormessage)
{
  const unsigned int type = OSL_TEXTURE_HANDLE_TYPE(texture_handle);
  const unsigned int slot = OSL_TEXTURE_HANDLE_SLOT(texture_handle);

  switch (type) {
    case OSL_TEXTURE_HANDLE_TYPE_SVM: {
      const float4 rgba = kernel_tex_image_interp_3d(nullptr, slot, *P, INTERPOLATION_NONE);
      if (nchannels > 0) {
        result[0] = rgba.x;
      }
      if (nchannels > 1) {
        result[1] = rgba.y;
      }
      if (nchannels > 2) {
        result[2] = rgba.z;
      }
      if (alpha) {
        *alpha = rgba.w;
      }
      return true;
    }
    default: {
      return false;
    }
  }
}

ccl_device_extern bool osl_environment(ccl_private ShaderGlobals *sg,
                                       DeviceString filename,
                                       ccl_private void *texture_handle,
                                       ccl_private OSLTextureOptions *opt,
                                       const ccl_private float3 *R,
                                       const ccl_private float3 *dRdx,
                                       const ccl_private float3 *dRdy,
                                       const int nchannels,
                                       ccl_private float *result,
                                       ccl_private float *dresultds,
                                       ccl_private float *dresultdt,
                                       ccl_private float *alpha,
                                       ccl_private float *dalphax,
                                       ccl_private float *dalphay,
                                       ccl_private void *errormessage)
{
  if (nchannels > 0) {
    result[0] = 1.0f;
  }
  if (nchannels > 1) {
    result[1] = 0.0f;
  }
  if (nchannels > 2) {
    result[2] = 1.0f;
  }
  if (alpha) {
    *alpha = 1.0f;
  }

  return false;
}

ccl_device_extern bool osl_get_textureinfo(ccl_private ShaderGlobals *sg,
                                           DeviceString filename,
                                           ccl_private void *texture_handle,
                                           DeviceString dataname,
                                           const int basetype,
                                           const int arraylen,
                                           const int aggegrate,
                                           ccl_private void *data,
                                           ccl_private void *errormessage)
{
  return false;
}

ccl_device_extern bool osl_get_textureinfo_st(ccl_private ShaderGlobals *sg,
                                              DeviceString filename,
                                              ccl_private void *texture_handle,
                                              const float s,
                                              const float t,
                                              DeviceString dataname,
                                              const int basetype,
                                              const int arraylen,
                                              const int aggegrate,
                                              ccl_private void *data,
                                              ccl_private void *errormessage)
{
  return osl_get_textureinfo(
      sg, filename, texture_handle, dataname, basetype, arraylen, aggegrate, data, errormessage);
}

/* Standard library */

#define OSL_OP_IMPL_II(name, op) \
  ccl_device_extern int name##_ii(const int a) \
  { \
    return op(a); \
  }
#define OSL_OP_IMPL_IF(name, op) \
  ccl_device_extern int name##_if(const float a) \
  { \
    return op(a); \
  }
#define OSL_OP_IMPL_FF(name, op) \
  ccl_device_extern float name##_ff(const float a) \
  { \
    return op(a); \
  }
#define OSL_OP_IMPL_DFDF(name, op) \
  ccl_device_extern void name##_dfdf(ccl_private float *res, ccl_private const float *a) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i]); \
    } \
  }
#define OSL_OP_IMPL_DFDV(name, op) \
  ccl_device_extern void name##_dfdv(ccl_private float *res, ccl_private const float3 *a) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i]); \
    } \
  }
#define OSL_OP_IMPL_FV(name, op) \
  ccl_device_extern float name##_fv(ccl_private const float3 *a) \
  { \
    return op(*a); \
  }
#define OSL_OP_IMPL_VV(name, op) \
  ccl_device_extern void name##_vv(ccl_private float3 *res, ccl_private const float3 *a) \
  { \
    *res = op(*a); \
  }
#define OSL_OP_IMPL_VV_(name, op) \
  ccl_device_extern void name##_vv(ccl_private float3 *res, ccl_private const float3 *a) \
  { \
    res->x = op(a->x); \
    res->y = op(a->y); \
    res->z = op(a->z); \
  }
#define OSL_OP_IMPL_DVDV(name, op) \
  ccl_device_extern void name##_dvdv(ccl_private float3 *res, ccl_private const float3 *a) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i]); \
    } \
  }
#define OSL_OP_IMPL_DVDV_(name, op) \
  ccl_device_extern void name##_dvdv(ccl_private float3 *res, ccl_private const float3 *a) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[i].x); \
      res[i].y = op(a[i].y); \
      res[i].z = op(a[i].z); \
    } \
  }

#define OSL_OP_IMPL_III(name, op) \
  ccl_device_extern int name##_iii(const int a, const int b) \
  { \
    return op(a, b); \
  }
#define OSL_OP_IMPL_FFF(name, op) \
  ccl_device_extern float name##_fff(const float a, const float b) \
  { \
    return op(a, b); \
  }
#define OSL_OP_IMPL_FVV(name, op) \
  ccl_device_extern float name##_fvv(ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    return op(*a, *b); \
  }
#define OSL_OP_IMPL_DFFDF(name, op) \
  ccl_device_extern void name##_dffdf( \
      ccl_private float *res, const float a, ccl_private const float *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a, b[i]); \
    } \
  }
#define OSL_OP_IMPL_DFDFF(name, op) \
  ccl_device_extern void name##_dfdff( \
      ccl_private float *res, ccl_private const float *a, const float b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b); \
    } \
  }
#define OSL_OP_IMPL_DFDFDF(name, op) \
  ccl_device_extern void name##_dfdfdf( \
      ccl_private float *res, ccl_private const float *a, ccl_private const float *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[i]); \
    } \
  }
#define OSL_OP_IMPL_DFVDV(name, op) \
  ccl_device_extern void name##_dfvdv( \
      ccl_private float *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[0], b[i]); \
    } \
  }
#define OSL_OP_IMPL_DFDVV(name, op) \
  ccl_device_extern void name##_dfdvv( \
      ccl_private float *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[0]); \
    } \
  }
#define OSL_OP_IMPL_DFDVDV(name, op) \
  ccl_device_extern void name##_dfdvdv( \
      ccl_private float *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[i]); \
    } \
  }
#define OSL_OP_IMPL_VVF_(name, op) \
  ccl_device_extern void name##_vvf( \
      ccl_private float3 *res, ccl_private const float3 *a, const float b) \
  { \
    res->x = op(a->x, b); \
    res->y = op(a->y, b); \
    res->z = op(a->z, b); \
  }
#define OSL_OP_IMPL_VVV(name, op) \
  ccl_device_extern void name##_vvv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    *res = op(*a, *b); \
  }
#define OSL_OP_IMPL_VVV_(name, op) \
  ccl_device_extern void name##_vvv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    res->x = op(a->x, b->x); \
    res->y = op(a->y, b->y); \
    res->z = op(a->z, b->z); \
  }
#define OSL_OP_IMPL_DVVDF_(name, op) \
  ccl_device_extern void name##_dvvdf( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[0].x, b[i]); \
      res[i].y = op(a[0].y, b[i]); \
      res[i].z = op(a[0].z, b[i]); \
    } \
  }
#define OSL_OP_IMPL_DVDVF_(name, op) \
  ccl_device_extern void name##_dvdvf( \
      ccl_private float3 *res, ccl_private const float3 *a, const float b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[i].x, b); \
      res[i].y = op(a[i].y, b); \
      res[i].z = op(a[i].z, b); \
    } \
  }
#define OSL_OP_IMPL_DVVDV(name, op) \
  ccl_device_extern void name##_dvvdv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[0], b[i]); \
    } \
  }
#define OSL_OP_IMPL_DVVDV_(name, op) \
  ccl_device_extern void name##_dvvdv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[0].x, b[i].x); \
      res[i].y = op(a[0].y, b[i].y); \
      res[i].z = op(a[0].z, b[i].z); \
    } \
  }
#define OSL_OP_IMPL_DVDVV(name, op) \
  ccl_device_extern void name##_dvdvv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[0]); \
    } \
  }
#define OSL_OP_IMPL_DVDVV_(name, op) \
  ccl_device_extern void name##_dvdvv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[i].x, b[0].x); \
      res[i].y = op(a[i].y, b[0].y); \
      res[i].z = op(a[i].z, b[0].z); \
    } \
  }
#define OSL_OP_IMPL_DVDVDF_(name, op) \
  ccl_device_extern void name##_dvdvdf( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[i].x, b[i]); \
      res[i].y = op(a[i].y, b[i]); \
      res[i].z = op(a[i].z, b[i]); \
    } \
  }
#define OSL_OP_IMPL_DVDVDV(name, op) \
  ccl_device_extern void name##_dvdvdv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[i]); \
    } \
  }
#define OSL_OP_IMPL_DVDVDV_(name, op) \
  ccl_device_extern void name##_dvdvdv( \
      ccl_private float3 *res, ccl_private const float3 *a, ccl_private const float3 *b) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i].x = op(a[i].x, b[i].x); \
      res[i].y = op(a[i].y, b[i].y); \
      res[i].z = op(a[i].z, b[i].z); \
    } \
  }

#define OSL_OP_IMPL_FFFF(name, op) \
  ccl_device_extern float name##_ffff(const float a, const float b, float c) \
  { \
    return op(a, b, c); \
  }
#define OSL_OP_IMPL_DFFFDF(name, op) \
  ccl_device_extern void name##_dfffdf( \
      ccl_private float *res, const float a, float b, ccl_private const float *c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a, b, c[i]); \
    } \
  }
#define OSL_OP_IMPL_DFFDFF(name, op) \
  ccl_device_extern void name##_dffdff( \
      ccl_private float *res, const float a, ccl_private const float *b, const float c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a, b[i], c); \
    } \
  }
#define OSL_OP_IMPL_DFFDFDF(name, op) \
  ccl_device_extern void name##_dffdfdf(ccl_private float *res, \
                                        const float a, \
                                        ccl_private const float *b, \
                                        ccl_private const float *c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a, b[i], c[i]); \
    } \
  }

#define OSL_OP_IMPL_DFDFFF(name, op) \
  ccl_device_extern void name##_dfdfff( \
      ccl_private float *res, ccl_private const float *a, const float b, float c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b, c); \
    } \
  }
#define OSL_OP_IMPL_DFDFFDF(name, op) \
  ccl_device_extern void name##_dfdffdf(ccl_private float *res, \
                                        ccl_private const float *a, \
                                        const float b, \
                                        ccl_private const float *c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b, c[i]); \
    } \
  }
#define OSL_OP_IMPL_DFDFDFF(name, op) \
  ccl_device_extern void name##_dfdfdff(ccl_private float *res, \
                                        ccl_private const float *a, \
                                        ccl_private const float *b, \
                                        const float c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[i], c); \
    } \
  }
#define OSL_OP_IMPL_DFDFDFDF(name, op) \
  ccl_device_extern void name##_dfdfdfdf(ccl_private float *res, \
                                         ccl_private const float *a, \
                                         ccl_private const float *b, \
                                         ccl_private const float *c) \
  { \
    for (int i = 0; i < 3; ++i) { \
      res[i] = op(a[i], b[i], c[i]); \
    } \
  }

#define OSL_OP_IMPL_XX(name, op) \
  OSL_OP_IMPL_FF(name, op) \
  OSL_OP_IMPL_DFDF(name, op) \
  OSL_OP_IMPL_VV_(name, op) \
  OSL_OP_IMPL_DVDV_(name, op)

#define OSL_OP_IMPL_XXX(name, op) \
  OSL_OP_IMPL_FFF(name, op) \
  OSL_OP_IMPL_DFFDF(name, op) \
  OSL_OP_IMPL_DFDFF(name, op) \
  OSL_OP_IMPL_DFDFDF(name, op) \
  OSL_OP_IMPL_VVV_(name, op) \
  OSL_OP_IMPL_DVVDV_(name, op) \
  OSL_OP_IMPL_DVDVV_(name, op) \
  OSL_OP_IMPL_DVDVDV_(name, op)

OSL_OP_IMPL_XX(osl_acos, acosf)
OSL_OP_IMPL_XX(osl_asin, asinf)
OSL_OP_IMPL_XX(osl_atan, atanf)
OSL_OP_IMPL_XXX(osl_atan2, atan2f)
OSL_OP_IMPL_XX(osl_cos, cosf)
OSL_OP_IMPL_XX(osl_sin, sinf)
OSL_OP_IMPL_XX(osl_tan, tanf)
OSL_OP_IMPL_XX(osl_cosh, coshf)
OSL_OP_IMPL_XX(osl_sinh, sinhf)
OSL_OP_IMPL_XX(osl_tanh, tanhf)

ccl_device_forceinline int safe_divide(const int a, const int b)
{
  return (b != 0) ? a / b : 0;
}
ccl_device_forceinline int safe_modulo(const int a, const int b)
{
  return (b != 0) ? a % b : 0;
}

OSL_OP_IMPL_III(osl_safe_div, safe_divide)
OSL_OP_IMPL_FFF(osl_safe_div, safe_divide)
OSL_OP_IMPL_III(osl_safe_mod, safe_modulo)

ccl_device_extern void osl_sincos_fff(const float a, ccl_private float *b, ccl_private float *c)
{
  sincos(a, b, c);
}
ccl_device_extern void osl_sincos_dfdff(const ccl_private float *a,
                                        ccl_private float *b,
                                        ccl_private float *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i], b + i, c);
  }
}
ccl_device_extern void osl_sincos_dffdf(const ccl_private float *a,
                                        ccl_private float *b,
                                        ccl_private float *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i], b, c + i);
  }
}
ccl_device_extern void osl_sincos_dfdfdf(const ccl_private float *a,
                                         ccl_private float *b,
                                         ccl_private float *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i], b + i, c + i);
  }
}
ccl_device_extern void osl_sincos_vvv(const ccl_private float3 *a,
                                      ccl_private float3 *b,
                                      ccl_private float3 *c)
{
  sincos(a->x, &b->x, &c->x);
  sincos(a->y, &b->y, &c->y);
  sincos(a->z, &b->z, &c->z);
}
ccl_device_extern void osl_sincos_dvdvv(const ccl_private float3 *a,
                                        ccl_private float3 *b,
                                        ccl_private float3 *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i].x, &b[i].x, &c->x);
    sincos(a[i].y, &b[i].y, &c->y);
    sincos(a[i].z, &b[i].z, &c->z);
  }
}
ccl_device_extern void osl_sincos_dvvdv(const ccl_private float3 *a,
                                        ccl_private float3 *b,
                                        ccl_private float3 *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i].x, &b->x, &c[i].x);
    sincos(a[i].y, &b->y, &c[i].y);
    sincos(a[i].z, &b->z, &c[i].z);
  }
}
ccl_device_extern void osl_sincos_dvdvdv(const ccl_private float3 *a,
                                         ccl_private float3 *b,
                                         ccl_private float3 *c)
{
  for (int i = 0; i < 3; ++i) {
    sincos(a[i].x, &b[i].x, &c[i].x);
    sincos(a[i].y, &b[i].y, &c[i].y);
    sincos(a[i].z, &b[i].z, &c[i].z);
  }
}

OSL_OP_IMPL_XX(osl_log, logf)
OSL_OP_IMPL_XX(osl_log2, log2f)
OSL_OP_IMPL_XX(osl_log10, log10f)
OSL_OP_IMPL_XX(osl_exp, expf)
OSL_OP_IMPL_XX(osl_exp2, exp2f)
OSL_OP_IMPL_XX(osl_expm1, expm1f)
OSL_OP_IMPL_XX(osl_erf, erff)
OSL_OP_IMPL_XX(osl_erfc, erfcf)

OSL_OP_IMPL_XXX(osl_pow, safe_powf)
OSL_OP_IMPL_VVF_(osl_pow, safe_powf)
OSL_OP_IMPL_DVVDF_(osl_pow, safe_powf)
OSL_OP_IMPL_DVDVF_(osl_pow, safe_powf)
OSL_OP_IMPL_DVDVDF_(osl_pow, safe_powf)

OSL_OP_IMPL_XX(osl_sqrt, sqrtf)
OSL_OP_IMPL_XX(osl_inversesqrt, 1.0f / sqrtf)
OSL_OP_IMPL_XX(osl_cbrt, cbrtf)

OSL_OP_IMPL_FF(osl_logb, logbf)
OSL_OP_IMPL_VV_(osl_logb, logbf)

OSL_OP_IMPL_FF(osl_floor, floorf)
OSL_OP_IMPL_VV_(osl_floor, floorf)
OSL_OP_IMPL_FF(osl_ceil, ceilf)
OSL_OP_IMPL_VV_(osl_ceil, ceilf)
OSL_OP_IMPL_FF(osl_round, roundf)
OSL_OP_IMPL_VV_(osl_round, roundf)
OSL_OP_IMPL_FF(osl_trunc, truncf)
OSL_OP_IMPL_VV_(osl_trunc, truncf)

ccl_device_forceinline float step_impl(const float edge, const float x)
{
  return x < edge ? 0.0f : 1.0f;
}

OSL_OP_IMPL_FF(osl_sign, compatible_signf)
OSL_OP_IMPL_VV_(osl_sign, compatible_signf)
OSL_OP_IMPL_FFF(osl_step, step_impl)
OSL_OP_IMPL_VVV_(osl_step, step_impl)

OSL_OP_IMPL_IF(osl_isnan, isnan)
OSL_OP_IMPL_IF(osl_isinf, isinf)
OSL_OP_IMPL_IF(osl_isfinite, isfinite)

OSL_OP_IMPL_II(osl_abs, abs)
OSL_OP_IMPL_XX(osl_abs, fabsf)
OSL_OP_IMPL_II(osl_fabs, abs)
OSL_OP_IMPL_XX(osl_fabs, fabsf)
OSL_OP_IMPL_XXX(osl_fmod, safe_modulo)
OSL_OP_IMPL_VVF_(osl_fmod, safe_modulo)
OSL_OP_IMPL_DVVDF_(osl_fmod, safe_modulo)
OSL_OP_IMPL_DVDVF_(osl_fmod, safe_modulo)
OSL_OP_IMPL_DVDVDF_(osl_fmod, safe_modulo)

OSL_OP_IMPL_FFFF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFFFDF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFFDFF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFFDFDF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFDFFF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFDFFDF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFDFDFF(osl_smoothstep, smoothstep)
OSL_OP_IMPL_DFDFDFDF(osl_smoothstep, smoothstep)

OSL_OP_IMPL_FVV(osl_dot, dot)
OSL_OP_IMPL_DFDVV(osl_dot, dot)
OSL_OP_IMPL_DFVDV(osl_dot, dot)
OSL_OP_IMPL_DFDVDV(osl_dot, dot)
OSL_OP_IMPL_VVV(osl_cross, cross)
OSL_OP_IMPL_DVDVV(osl_cross, cross)
OSL_OP_IMPL_DVVDV(osl_cross, cross)
OSL_OP_IMPL_DVDVDV(osl_cross, cross)
OSL_OP_IMPL_FV(osl_length, len)
OSL_OP_IMPL_DFDV(osl_length, len)
OSL_OP_IMPL_FVV(osl_distance, distance)
OSL_OP_IMPL_DFDVV(osl_distance, distance)
OSL_OP_IMPL_DFVDV(osl_distance, distance)
OSL_OP_IMPL_DFDVDV(osl_distance, distance)
OSL_OP_IMPL_VV(osl_normalize, safe_normalize)
OSL_OP_IMPL_DVDV(osl_normalize, safe_normalize)

ccl_device_extern void osl_calculatenormal(ccl_private float3 *res,
                                           ccl_private ShaderGlobals *sg,
                                           const ccl_private float3 *p)
{
  if (sg->flipHandedness) {
    *res = cross(p[2], p[1]);
  }
  else {
    *res = cross(p[1], p[2]);
  }
}

ccl_device_extern float osl_area(const ccl_private float3 *p)
{
  return len(cross(p[2], p[1]));
}

ccl_device_extern float osl_filterwidth_fdf(const ccl_private float *x)
{
  return sqrtf(x[1] * x[1] + x[2] * x[2]);
}

ccl_device_extern void osl_filterwidth_vdv(ccl_private float *res, const ccl_private float *x)
{
  for (int i = 0; i < 3; ++i) {
    res[i] = osl_filterwidth_fdf(x + i);
  }
}

ccl_device_extern bool osl_raytype_bit(ccl_private ShaderGlobals *sg, const int bit)
{
  return (sg->raytype & bit) != 0;
}
