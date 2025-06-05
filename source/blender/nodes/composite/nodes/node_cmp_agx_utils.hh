#pragma once

namespace blender::nodes::node_composite_agx_view_transform_cc {

/* ##########################################################################
    Custom Structs
    ---------------------------
*/

// A struct holding 3x3 matrix
struct float3x3 {
  float3 x;
  float3 y;
  float3 z;
};

// A struct holding xy for R, G, B, and White point.
struct Chromaticities {
  float2 red;
  float2 green;
  float2 blue;
  float2 white;
};

/* ##########################################################################
    Color Spaces Coordinates
    ---------------------------
*/

// AP0 white point changed to D65 for simplicity
static const Chromaticities AP0_PRI = {
    /* r: */ {0.734771f, 0.264663f},
    /* g: */ {-0.00795f, 1.006817f},
    /* b: */ {0.016895f, -0.062809f},
    /* w: */ {0.312727f,   0.329023f}
};

// AP1 white point changed to D65 for simplicity
static const Chromaticities AP1_PRI = {
    /* r: */ {0.713016f, 0.292962f},
    /* g: */ {0.158021f, 0.835539f},
    /* b: */ {0.129469f, 0.045065f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities P3D65_PRI = {
    /* r: */ {0.68f, 0.32f},
    /* g: */ {0.265f, 0.69f},
    /* b: */ {0.15f, 0.06f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities REC709_PRI = {
    /* r: */ {0.64f, 0.33f},
    /* g: */ {0.3f, 0.6f},
    /* b: */ {0.15f, 0.06f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities REC2020_PRI = {
    /* r: */ {0.708f, 0.292f},
    /* g: */ {0.17f, 0.797f},
    /* b: */ {0.131f, 0.046f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities AWG3_PRI = {
    /* r: */ {0.684f, 0.313f},
    /* g: */ {0.221f, 0.848f},
    /* b: */ {0.0861f, -0.102f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities AWG4_PRI = {
    /* r: */ {0.7347f,  0.2653f},
    /* g: */ {0.1424f, 0.8576f},
    /* b: */ {0.0991f,-0.0308f},
    /* w: */ {0.312727f, 0.329023f}
};

static const Chromaticities EGAMUT_PRI = {
    /* r: */ {0.8f,  0.3177f},
    /* g: */ {0.18f, 0.9f},
    /* b: */ {0.065f, -0.0805f},
    /* w: */ {0.312727f, 0.329023f}
};

/* ##########################################################################
    Other Constants
    ---------------------------
*/

static inline float3x3 identity_mtx = {
  {1.0f, 0.0f, 0.0f},
  {0.0f, 1.0f, 0.0f},
  {0.0f, 0.0f, 1.0f}
};

static const float pi = 3.14159265358979323846f;

// Create a const array of Chromaticities for enum's index-based selection
static const Chromaticities COLOR_SPACE_PRI[] = {
    AP0_PRI,
    AP1_PRI,
    P3D65_PRI,
    REC709_PRI,
    REC2020_PRI,
    AWG3_PRI,
    AWG4_PRI,
    EGAMUT_PRI
};

/* ##########################################################################
    Functions
    ---------------------------------
*/

// helper functions to convert between radians and degrees
static inline float radians(float d) {return d * (pi / 180.0f);}
static inline float degrees(float r) {return r * (180.0f / pi);}

// Helper function to create a float2
static inline float2 make_float2(const float x, const float y) {
  return float2{x, y};
}

// Helper function to create a float3
static inline float3 make_float3(float x, float y, float z) {
  return float3{x, y, z};
}

// Helper function to create a float3x3
static inline float3x3 make_float3x3(float3 a, float3 b, float3 c) {
  float3x3 d;
  d.x = a, d.y = b, d.z = c;
  return d;
}

static inline Chromaticities make_chromaticities( float2 A, float2 B, float2 C, float2 D) {
  Chromaticities E;
  E.red = A; E.green = B; E.blue = C; E.white = D;
  return E;
}

// Multiply float3 vector a and 3x3 matrix m
static inline float3 mult_f3_f33(float3 a, float3x3 m) {
  return make_float3(
    m.x.x * a.x + m.x.y * a.y + m.x.z * a.z,
    m.y.x * a.x + m.y.y * a.y + m.y.z * a.z,
    m.z.x * a.x + m.z.y * a.y + m.z.z * a.z
  );
}

// Calculate inverse of 3x3 matrix
static inline float3x3 inv_f33(float3x3 m) {
  float d = m.x.x * (m.y.y * m.z.z - m.z.y * m.y.z) -
            m.x.y * (m.y.x * m.z.z - m.y.z * m.z.x) +
            m.x.z * (m.y.x * m.z.y - m.y.y * m.z.x);
  float id = 1.0f / d;
  float3x3 c = identity_mtx;
  c.x.x = id * (m.y.y * m.z.z - m.z.y * m.y.z);
  c.x.y = id * (m.x.z * m.z.y - m.x.y * m.z.z);
  c.x.z = id * (m.x.y * m.y.z - m.x.z * m.y.y);
  c.y.x = id * (m.y.z * m.z.x - m.y.x * m.z.z);
  c.y.y = id * (m.x.x * m.z.z - m.x.z * m.z.x);
  c.y.z = id * (m.y.x * m.x.z - m.x.x * m.y.z);
  c.z.x = id * (m.y.x * m.z.y - m.z.x * m.y.y);
  c.z.y = id * (m.z.x * m.x.y - m.x.x * m.z.y);
  c.z.z = id * (m.x.x * m.y.y - m.y.x * m.x.y);
  return c;
}

static inline float3 clampf3(float3 a, float mn, float mx) {
  return float3{
    math::clamp(a.x, mn, mx),
    math::clamp(a.y, mn, mx),
    math::clamp(a.z, mn, mx)
  };
}

static inline float3 maxf3(float b, float3 a) {
  // For each component of float3 a, return max of component and float b
  return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

static inline float3 minf3(float b, float3 a) {
  // For each component of float3 a, return min of component and float b
  return make_float3(fminf(a.x, b), fminf(a.y, b), fminf(a.z, b));
}

static inline float3 powf3(float3 a, float b) {
  // Raise each component of float3 a to power b
  return make_float3(pow(a.x, b), pow(a.y, b), pow(a.z, b));
}

static inline float3 log2f3(float3 RGB) {
  return make_float3(log2f(RGB.x), log2f(RGB.y), log2f(RGB.z));
}

static inline float sign(float x) {
  // Return the sign of float x
  if (x > 0.0f) return 1.0f;
  if (x < 0.0f) return -1.0f;
  return 0.0f;
}

static inline float spowf(float a, float b) {
  // Compute "safe" power of float a, reflected over the origin

  a=sign(a)*pow(fabsf(a), b);
  return a;
}

static inline float3 spowf3(float3 a, float b) {
  // Compute "safe" power of float3 a, reflected over the origin
  return make_float3(
    sign(a.x)*pow(fabsf(a.x), b),
    sign(a.y)*pow(fabsf(a.y), b),
    sign(a.z)*pow(fabsf(a.z), b)
  );
}

static inline float log10f(float x) {
  return log(x) * (1.0f / log(10.0f));
}

static inline float3 log2lin(float3 rgb, int tf, float generic_log2_min_expo = -10, float generic_log2_max_expo = 6.5) {
  if (tf == 0) return rgb;
  else if (tf == 1) { // ACEScct
    rgb.x = rgb.x > 0.155251141552511f ? pow(2.0f, rgb.x * 17.52f - 9.72f) : (rgb.x - 0.0729055341958355f) / 10.5402377416545f;
    rgb.y = rgb.y > 0.155251141552511f ? pow(2.0f, rgb.y * 17.52f - 9.72f) : (rgb.y - 0.0729055341958355f) / 10.5402377416545f;
    rgb.z = rgb.z > 0.155251141552511f ? pow(2.0f, rgb.z * 17.52f - 9.72f) : (rgb.z - 0.0729055341958355f) / 10.5402377416545f;
  } else if (tf == 2) { // Arri LogC3 EI 800
    rgb.x = rgb.x > 0.149658f ? (pow(10.0f, (rgb.x - 0.385537f) / 0.24719f) - 0.052272f) / 5.555556f : (rgb.x - 0.092809f) / 5.367655f;
    rgb.y = rgb.y > 0.149658f ? (pow(10.0f, (rgb.y - 0.385537f) / 0.24719f) - 0.052272f) / 5.555556f : (rgb.y - 0.092809f) / 5.367655f;
    rgb.z = rgb.z > 0.149658f ? (pow(10.0f, (rgb.z - 0.385537f) / 0.24719f) - 0.052272f) / 5.555556f : (rgb.z - 0.092809f) / 5.367655f;
  } else if (tf == 3){  // Arri LogC 4
    const float a = (pow(2.0f, 18.0f) - 16.0f) / 117.45f;
    const float b = (1023.0f - 95.0f) / 1023.0f;
    const float c = 95.0f / 1023.f;
    const float s = (7.f * logf(2.0f) * pow(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
    const float t = (pow(2.0f, 14.0f * ((-1.0f * c) / b) + 6.0f) - 64.0f) / a;

    rgb.x = rgb.x < 0.0f ? rgb.x * s + t : (pow(2.0f, (14.0f * (rgb.x - c) / b + 6.0f)) - 64.0f) / a;
    rgb.y = rgb.y < 0.0f ? rgb.y * s + t : (pow(2.0f, (14.0f * (rgb.y - c) / b + 6.0f)) - 64.0f) / a;
    rgb.z = rgb.z < 0.0f ? rgb.z * s + t : (pow(2.0f, (14.0f * (rgb.z - c) / b + 6.0f)) - 64.0f) / a;
  } else if (tf == 4){ // User controlled PureLog2
    float mx = generic_log2_max_expo;
    float mn = generic_log2_min_expo;

    rgb.x = 0.18*pow(2,(rgb.x*(mx-mn)+mn));
    rgb.y = 0.18*pow(2,(rgb.y*(mx-mn)+mn));
    rgb.z = 0.18*pow(2,(rgb.z*(mx-mn)+mn));
  }
  return rgb;
}

static inline float3 lin2log(float3 rgb, int tf, float generic_log2_min_expo = -10, float generic_log2_max_expo = 6.5) {
  if (tf == 0) return rgb;
  else if (tf == 1) { // ACEScct
    rgb.x = rgb.x > 0.0078125f ? (log2f(rgb.x) + 9.72f) / 17.52f : 10.5402377416545f * rgb.x + 0.0729055341958355f;
    rgb.y = rgb.y > 0.0078125f ? (log2f(rgb.y) + 9.72f) / 17.52f : 10.5402377416545f * rgb.y + 0.0729055341958355f;
    rgb.z = rgb.z > 0.0078125f ? (log2f(rgb.z) + 9.72f) / 17.52f : 10.5402377416545f * rgb.z + 0.0729055341958355f;
  } else if (tf == 2) { // Arri LogC3 EI 800
    rgb.x = rgb.x > 0.010591f ? 0.24719f * log10f(5.555556f * rgb.x + 0.052272f) + 0.385537f : 5.367655f * rgb.x + 0.092809f;
    rgb.y = rgb.y > 0.010591f ? 0.24719f * log10f(5.555556f * rgb.y + 0.052272f) + 0.385537f : 5.367655f * rgb.y + 0.092809f;
    rgb.z = rgb.z > 0.010591f ? 0.24719f * log10f(5.555556f * rgb.z + 0.052272f) + 0.385537f : 5.367655f * rgb.z + 0.092809f;
  } else if (tf == 3){  // Arri LogC 4
    const float a = (pow(2.0f, 18.0f) - 16.0f) / 117.45f;
    const float b = (1023.0f - 95.0f) / 1023.0f;
    const float c = 95.0f / 1023.f;
    const float s = (7.f * logf(2.0f) * pow(2.0f, 7.0f - 14.0f * c / b)) / (a * b);
    const float t = (pow(2.0f, 14.0f * ((-1.0f * c) / b) + 6.0f) - 64.0f) / a;

    rgb.x = rgb.x >= t ? ((log2f(a * rgb.x + 64.f) - 6.f) / 14.f) * b + c : (rgb.x - t) / s;
    rgb.y = rgb.y >= t ? ((log2f(a * rgb.y + 64.f) - 6.f) / 14.f) * b + c : (rgb.y - t) / s;
    rgb.z = rgb.z >= t ? ((log2f(a * rgb.z + 64.f) - 6.f) / 14.f) * b + c : (rgb.z - t) / s;
  } else if (tf == 4) { // User controlled PureLog2
    rgb = log2f3(rgb / 0.18f);
    rgb = clampf3(rgb, generic_log2_min_expo, generic_log2_max_expo);

    rgb = (rgb + fabsf(generic_log2_min_expo)) / (fabsf(generic_log2_min_expo)+fabsf(generic_log2_max_expo));
  }
  return rgb;
}

static inline float sigmoid(float in, float sp, float tp, float Pslope, float px, float py,float s0=1.0f,float t0=0.0f)
{
  //calculate Shoulder;
  float ss =spowf(((spowf((Pslope*((s0-px)/(1-py))),sp)-1)*(spowf(Pslope*(s0-px),-sp))),-1/sp);
  float ms = Pslope*(in-px)/ss;
  float fs = ms/spowf(1+(spowf(ms,sp)),1/sp);

  //calculate Toe
  float ts =spowf(((spowf((Pslope*((px-t0)/(py))),tp)-1)*(spowf(Pslope*(px-t0),-tp))),-1/tp);
  float mr = (Pslope*(in-px))/-ts;
  float ft = mr/spowf(1+(spowf(mr,tp)),1/tp);

  in = in >= px ? ss * fs + py : (-ts * ft) + py;

  return in;
}

static inline float3x3 RGBtoXYZ( Chromaticities N) {
  float3x3 M = make_float3x3(
    make_float3(N.red.x/N.red.y, N.green.x / N.green.y, N.blue.x / N.blue.y),
    make_float3(1.0, 1.0, 1.0),
    make_float3(
      (1-N.red.x-N.red.y) / N.red.y, (1-N.green.x-N.green.y) / N.green.y, (1-N.blue.x-N.blue.y)/N.blue.y
    )
  );
  float3 wh = make_float3(
    N.white.x / N.white.y, 1.0, (1-N.white.x-N.white.y) / N.white.y
  );
  wh = mult_f3_f33(wh, inv_f33(M));
  M = make_float3x3(
    make_float3(M.x.x*wh.x , M.x.y*wh.y , M.x.z*wh.z),
    make_float3(M.y.x*wh.x, M.y.y*wh.y, M.y.z*wh.z),
    make_float3(M.z.x*wh.x,M.z.y*wh.y,M.z.z*wh.z)
  );
  return M;
}

static inline float3x3 XYZtoRGB( Chromaticities N) {
  float3x3 M = inv_f33(RGBtoXYZ(N));
  return M;
}

static inline float3x3 transpose_f33( float3x3 A) {
  float3x3 B = A;
  A.x=make_float3(B.x.x,B.y.x,B.z.x);
  A.y=make_float3(B.x.y,B.y.y,B.z.y);
  A.z=make_float3(B.x.z,B.y.z,B.z.z);

  return A;
}

static inline float3x3 mult_f33_f33( float3x3 A, float3x3 B) {
  A = transpose_f33(A);
  float3x3 C = B;
  B.x= mult_f3_f33(A.x,C);
  B.y= mult_f3_f33(A.y,C);
  B.z= mult_f3_f33(A.z,C);
  B = transpose_f33(B);

  return B;
}

static inline float3x3 RGBtoRGB(Chromaticities N,Chromaticities M){
  float3x3 In2XYZ = RGBtoXYZ(N);
  float3x3 XYZ2Out = XYZtoRGB(M);

  float3x3 rgbtorgb = mult_f33_f33(In2XYZ,XYZ2Out);

  return rgbtorgb;
}

static inline float lerp_chromaticity_angle(float h1, float h2, float t) {
    float delta = h2 - h1;
    if (delta > 0.5f) delta -= 1.0f;
    else if (delta < -0.5f) delta += 1.0f;
    float lerped = h1 + t * delta;
    return lerped - floorf(lerped);
}

static inline float3 compensate_low_side(float3 rgb, bool use_hacky_lerp, Chromaticities working_chrom) {
    // Hardcoded Rec.2020 luminance coefficients (2015 CMFs)
    const float3 luminance_coeffs = make_float3(0.265818f, 0.59846986f, 0.1357121f);
    Chromaticities rec2020 = REC2020_PRI;

    // Convert RGB to Rec.2020 for luminance calculation
    float3x3 working_to_rec2020 = RGBtoRGB(working_chrom, rec2020);
    float3 rgb_rec2020 = mult_f3_f33(rgb, working_to_rec2020);

    // Calculate original luminance Y
    float Y = rgb_rec2020.x * luminance_coeffs.x +
              rgb_rec2020.y * luminance_coeffs.y +
              rgb_rec2020.z * luminance_coeffs.z;

    // Calculate inverse RGB in working space
    float max_rgb = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    float3 inverse_rgb = make_float3(max_rgb - rgb.x, max_rgb - rgb.y, max_rgb - rgb.z);

    // Calculate max of the inverse
    float max_inv_rgb = fmaxf(inverse_rgb.x, fmaxf(inverse_rgb.y, inverse_rgb.z));

    // Convert inverse RGB to Rec.2020 for Y calculation
    float3 inverse_rec2020 = mult_f3_f33(inverse_rgb, working_to_rec2020);
    float Y_inverse = inverse_rec2020.x * luminance_coeffs.x +
                      inverse_rec2020.y * luminance_coeffs.y +
                      inverse_rec2020.z * luminance_coeffs.z;

    // Calculate compensation values
    float y_compensate = (max_inv_rgb - Y_inverse + Y);
    if (use_hacky_lerp) {
        float Y_clipped = math::clamp(pow(Y, 0.08f), 0.0f, 1.0f);
        y_compensate = y_compensate + Y_clipped * (Y - y_compensate);
    }

    // Offset to avoid negatives
    float min_rgb = fminf(rgb.x, fminf(rgb.y, rgb.z));
    float offset = fmaxf(-min_rgb, 0.0f);
    float3 rgb_offset = make_float3(rgb.x + offset, rgb.y + offset, rgb.z + offset);

    // Calculate max of the offseted RGB
    float max_offset = fmaxf(rgb_offset.x, fmaxf(rgb_offset.y, rgb_offset.z));

    // Calculate new luminance after offset
    float3 offset_rec2020 = mult_f3_f33(rgb_offset, working_to_rec2020);
    float Y_new = offset_rec2020.x * luminance_coeffs.x +
                  offset_rec2020.y * luminance_coeffs.y +
                  offset_rec2020.z * luminance_coeffs.z;

    // Calculate the inverted RGB offset
    float3 inverse_offset = make_float3(max_offset - rgb_offset.x,
                                       max_offset - rgb_offset.y,
                                       max_offset - rgb_offset.z);

    // Calculate max of the inverse
    float max_inv_offset = fmaxf(inverse_offset.x, fmaxf(inverse_offset.y, inverse_offset.z));

    float3 inverse_offset_rec2020 = mult_f3_f33(inverse_offset, working_to_rec2020);
    float Y_inverse_offset = inverse_offset_rec2020.x * luminance_coeffs.x +
                             inverse_offset_rec2020.y * luminance_coeffs.y +
                             inverse_offset_rec2020.z * luminance_coeffs.z;

    float Y_new_compensate = (max_inv_offset - Y_inverse_offset + Y_new);
    if (use_hacky_lerp) {
        float Y_new_clipped = math::clamp(pow(Y_new, 0.08f), 0.0f, 1.0f);
        Y_new_compensate = Y_new_compensate + Y_new_clipped * (Y_new - Y_new_compensate);
    }

    // Adjust luminance ratio
    float ratio = (Y_new_compensate > y_compensate) ? (y_compensate / Y_new_compensate) : 1.0f;
    return make_float3(rgb_offset.x * ratio, rgb_offset.y * ratio, rgb_offset.z * ratio);
}

static inline Chromaticities CenterPrimaries(Chromaticities N){
  N.red.x = N.red.x-N.white.x;
  N.red.y = N.red.y-N.white.y;
  N.green.x = N.green.x-N.white.x;
  N.green.y = N.green.y-N.white.y;
  N.blue.x = N.blue.x-N.white.x;
  N.blue.y = N.blue.y-N.white.y;

  return N;
}

static inline Chromaticities DeCenterPrimaries(Chromaticities N){
  N.red.x = N.red.x+N.white.x;
  N.red.y = N.red.y+N.white.y;
  N.green.x = N.green.x+N.white.x;
  N.green.y = N.green.y+N.white.y;
  N.blue.x = N.blue.x+N.white.x;
  N.blue.y = N.blue.y+N.white.y;

  return N;
}

static inline float2 cartesian_to_polar2(float2 a) {
  float2 b = a;
  b.y = atan2(a.y,a.x);

  return make_float2(sqrt(a.x*a.x+ a.y*a.y),b.y);
}

static inline float2 polar_to_cartesian2(float2 a) {

  return make_float2(a.x * cos(a.y), a.x * sin(a.y));
}

static inline Chromaticities RotatePrimary(Chromaticities N,float rrot,float grot,float brot){
  //rotatation parameter excepted in degrees, but internally transformed to radians

  N = CenterPrimaries(N);
  N.red = cartesian_to_polar2(N.red);
  N.green = cartesian_to_polar2(N.green);
  N.blue = cartesian_to_polar2(N.blue);

  rrot = radians(rrot);
  grot = radians(grot);
  brot = radians(brot);

  N.red.y = N.red.y + rrot;
  N.green.y = N.green.y + grot;
  N.blue.y = N.blue.y + brot;

  N.red = polar_to_cartesian2(N.red);
  N.green = polar_to_cartesian2(N.green);
  N.blue = polar_to_cartesian2(N.blue);

  N = DeCenterPrimaries(N);

  return N;
}

static inline Chromaticities ScalePrim(Chromaticities N,float rs,float gs,float bs){
  N = CenterPrimaries(N);

  N.red = make_float2(N.red.x*rs,N.red.y*rs);
  N.green = make_float2(N.green.x*gs,N.green.y*gs);
  N.blue = make_float2(N.blue.x*bs,N.blue.y*bs);

  N = DeCenterPrimaries(N);

  return N;
}

static inline float2 Line_equation(float2 a, float2 b)
{
    float dx = b.x - a.x;
    if (fabsf(dx) < 1e-6f) {
        /* vertical line → slope = +∞, intercept = x */
        const float kInf = 1e30f * 1e30f;
        return make_float2(kInf, a.x);
    }
    float m = (b.y - a.y) / dx;
    float c = a.y - m * a.x;
    return make_float2(m, c);
}

static inline Chromaticities Polygon(Chromaticities N){
  Chromaticities M = N;

  N.red = Line_equation(M.red,M.green);
  N.green = Line_equation(M.red,M.blue);
  N.blue = Line_equation(M.blue,M.green);

  return N;
}

static inline Chromaticities PrimariesLines(Chromaticities N){
    Chromaticities M = N;

    N.red = Line_equation(M.red,M.white);
    N.green = Line_equation(M.green,M.white);
    N.blue = Line_equation(M.blue,M.white);

    return N;
}

static inline float2 intersection(float2 l1, float2 l2) {
  float m1 = l1.x, c1 = l1.y;
  float m2 = l2.x, c2 = l2.y;

  bool inf1 = isfinite(m1) == false;
  bool inf2 = isfinite(m2) == false;

  float x, y;
  if (inf1 && inf2) {
    // two verticals (parallel) → just default (0,0)
    x = y = 0.0f;
  }
  else if (inf1) {
    // first is vertical: x = c1
    x = c1;
    y = m2 * x + c2;
  }
  else if (inf2) {
    // second is vertical: x = c2
    x = c2;
    y = m1 * x + c1;
  }
  else {
    x = (c2 - c1) / (m1 - m2);
    y = m1 * x + c1;
  }
  return make_float2(x, y);
}

static inline Chromaticities InsetPrimaries(Chromaticities N, float cpr, float cpg, float cpb, float ored, float og, float ob, float achromatic_rotate = 0, float achromatic_outset = 0) {
  Chromaticities M = N;

  Chromaticities original_N = N;
  Chromaticities scaled_N = ScalePrim(N, 4.0, 4.0, 4.0);

  N = RotatePrimary(scaled_N, ored, og, ob);

  M = Polygon(M);

  float2 redline = ored > 0 ? M.red : M.green;
  float2 greenline = og > 0 ? M.blue : M.red;
  float2 blueline = ob > 0 ? M.green : M.blue;

  // compute the line eqns from each rotated‐primary → whitepoint
  float2 wp = original_N.white;
  float2 lr = Line_equation( make_float2(N.red.x,   N.red.y),   wp );
  float2 lg = Line_equation( make_float2(N.green.x, N.green.y), wp );
  float2 lb = Line_equation( make_float2(N.blue.x,  N.blue.y),  wp );

  // intersect each with the chosen triangle edge
  float2 Pr = intersection(lr, redline);
  float2 Pg = intersection(lg, greenline);
  float2 Pb = intersection(lb, blueline);

  // assign back into N
  N.red   = Pr;
  N.green = Pg;
  N.blue  = Pb;

  cpr = 1 - cpr;
  cpg = 1 - cpg;
  cpb = 1 - cpb;

  N = ScalePrim(N, cpr, cpg, cpb);

  // --- Start of Achromatic Tinting ---
  float2 original_white = Polygon(original_N).white;
  const float arbitrary_scale = 4.0f;

  // Scale & rotate the achromatic point
  float2 scaled_achromatic = make_float2(
      original_white.x,
      original_white.y * arbitrary_scale
  );
  float achromatic_rotate_radians = radians(achromatic_rotate);
  float dx = scaled_achromatic.x - original_white.x;
  float dy = scaled_achromatic.y - original_white.y;
  float2 rotated_achromatic = make_float2(
      original_white.x + dx * cos(achromatic_rotate_radians) - dy * sin(achromatic_rotate_radians),
      original_white.y + dx * sin(achromatic_rotate_radians) + dy * cos(achromatic_rotate_radians)
  );

  // Build the infinite achromatic ray and triangle edges
  float2 la = Line_equation(rotated_achromatic, original_white);
  float2 e1 = Line_equation(original_N.red,   original_N.green);
  float2 e2 = Line_equation(original_N.green, original_N.blue);
  float2 e3 = Line_equation(original_N.blue,  original_N.red);

  // Find their intersections
  float2 i1 = intersection(la, e1);
  float2 i2 = intersection(la, e2);
  float2 i3 = intersection(la, e3);

  // Pick the first intersect that lies on both the achromatic ray segment (t∈[0,1])
  // and the edge segment (u∈[0,1]), else fallback to white
  float2 hull_achromatic = original_white;
  float2 A = rotated_achromatic, B = original_white;
  auto on_segment = [&](float2 P, float2 C, float2 D){
    float t = (fabsf(B.x-A.x)>fabsf(B.y-A.y))
      ? (P.x-A.x)/(B.x-A.x)
      : (P.y-A.y)/(B.y-A.y);
    float u = (fabsf(D.x-C.x)>fabsf(D.y-C.y))
      ? (P.x-C.x)/(D.x-C.x)
      : (P.y-C.y)/(D.y-C.y);
    return (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f);
  };
  if      (on_segment(i1, original_N.red,   original_N.green)) hull_achromatic = i1;
  else if (on_segment(i2, original_N.green, original_N.blue )) hull_achromatic = i2;
  else if (on_segment(i3, original_N.blue,  original_N.red  )) hull_achromatic = i3;

  // Move whitepoint towards hull_achromatic by achromatic_outset
  float2 interp = make_float2(
      (original_white.x - hull_achromatic.x) * (1.0f - achromatic_outset),
      (original_white.y - hull_achromatic.y) * (1.0f - achromatic_outset)
  );
  N.white.x = hull_achromatic.x + interp.x;
  N.white.y = hull_achromatic.y + interp.y;

  return N;
}

}  // namespace blender::nodes::node_composite_agx_view_transform_cc
