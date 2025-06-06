/* SPDX-FileCopyrightText: 2024 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/* Blur the input horizontally by applying a fourth order IIR filter approximating a Gaussian
 * filter using Van Vliet's design method. This is based on the following paper:
 *
 *   Van Vliet, Lucas J., Ian T. Young, and Piet W. Verbeek. "Recursive Gaussian derivative
 *   filters." Proceedings. Fourteenth International Conference on Pattern Recognition (Cat. No.
 *   98EX170). Vol. 1. IEEE, 1998.
 *
 * We decomposed the filter into two second order filters, so we actually run four filters per row
 * in parallel, one for the first causal filter, one for the first non causal filter, one for the
 * second causal filter, and one for the second non causal filter, storing the result of each
 * separately. See the VanVlietGaussianCoefficients class and the implementation for more
 * information. */

#include "gpu_shader_compositor_texture_utilities.glsl"

#define FILTER_ORDER 2

void main()
{
  /* The shader runs parallel across rows but serially across columns. */
  int y = int(gl_GlobalInvocationID.x);
  int width = texture_size(input_tx).x;

  /* The second dispatch dimension is four dispatches:
   *
   *   0 -> First causal filter.
   *   1 -> First non causal filter.
   *   2 -> Second causal filter.
   *   3 -> Second non causal filter.
   *
   * We detect causality by even numbers. */
  bool is_causal = gl_GlobalInvocationID.y % 2 == 0;
  float2 first_feedforward_coefficients = is_causal ? first_causal_feedforward_coefficients :
                                                      first_non_causal_feedforward_coefficients;
  float first_boundary_coefficient = is_causal ? first_causal_boundary_coefficient :
                                                 first_non_causal_boundary_coefficient;
  float2 second_feedforward_coefficients = is_causal ? second_causal_feedforward_coefficients :
                                                       second_non_causal_feedforward_coefficients;
  float second_boundary_coefficient = is_causal ? second_causal_boundary_coefficient :
                                                  second_non_causal_boundary_coefficient;
  /* And we detect the filter by order. */
  bool is_first_filter = gl_GlobalInvocationID.y < 2;
  float2 feedforward_coefficients = is_first_filter ? first_feedforward_coefficients :
                                                      second_feedforward_coefficients;
  float2 feedback_coefficients = is_first_filter ? first_feedback_coefficients :
                                                   second_feedback_coefficients;
  float boundary_coefficient = is_first_filter ? first_boundary_coefficient :
                                                 second_boundary_coefficient;

  /* Create an array that holds the last FILTER_ORDER inputs along with the current input. The
   * current input is at index 0 and the oldest input is at index FILTER_ORDER. We assume Neumann
   * boundary condition, so we initialize all inputs by the boundary pixel. */
  int2 boundary_texel = is_causal ? int2(0, y) : int2(width - 1, y);
  float4 input_boundary = texture_load(input_tx, boundary_texel);
  float4 inputs[FILTER_ORDER + 1] = float4_array(input_boundary, input_boundary, input_boundary);

  /* Create an array that holds the last FILTER_ORDER outputs along with the current output. The
   * current output is at index 0 and the oldest output is at index FILTER_ORDER. We assume Neumann
   * boundary condition, so we initialize all outputs by the boundary pixel multiplied by the
   * boundary coefficient. See the VanVlietGaussianCoefficients class for more information on the
   * boundary handing. */
  float4 output_boundary = input_boundary * boundary_coefficient;
  float4 outputs[FILTER_ORDER + 1] = float4_array(
      output_boundary, output_boundary, output_boundary);

  for (int x = 0; x < width; x++) {
    /* Run forward across rows for the causal filter and backward for the non causal filter. */
    int2 texel = is_causal ? int2(x, y) : int2(width - 1 - x, y);
    inputs[0] = texture_load(input_tx, texel);

    /* Compute the filter based on its difference equation, this is not in the Van Vliet paper
     * because the filter was decomposed, but it is essentially similar to Equation (28) for the
     * causal filter or Equation (29) for the non causal filter in Deriche's paper, except it is
     * second order, not fourth order.
     *
     *   Deriche, Rachid. Recursively implementating the Gaussian and its derivatives. Diss. INRIA,
     *   1993.
     *
     * The only difference is that the non causal filter ignores the current value and starts from
     * the previous input, as can be seen in the subscript of the first input term in both
     * equations. So add one while indexing the non causal inputs. */
    outputs[0] = float4(0.0f);
    int first_input_index = is_causal ? 0 : 1;
    for (int i = 0; i < FILTER_ORDER; i++) {
      outputs[0] += feedforward_coefficients[i] * inputs[first_input_index + i];
      outputs[0] -= feedback_coefficients[i] * outputs[i + 1];
    }

    /* Store the causal and non causal outputs of each of the two filters independently, then sum
     * them in a separate shader dispatch for better parallelism. */
    if (is_causal) {
      if (is_first_filter) {
        imageStore(first_causal_output_img, texel, outputs[0]);
      }
      else {
        imageStore(second_causal_output_img, texel, outputs[0]);
      }
    }
    else {
      if (is_first_filter) {
        imageStore(first_non_causal_output_img, texel, outputs[0]);
      }
      else {
        imageStore(second_non_causal_output_img, texel, outputs[0]);
      }
    }

    /* Shift the inputs temporally by one. The oldest input is discarded, while the current input
     * will retain its value but will be overwritten with the new current value in the next
     * iteration. */
    for (int i = FILTER_ORDER; i >= 1; i--) {
      inputs[i] = inputs[i - 1];
    }

    /* Shift the outputs temporally by one. The oldest output is discarded, while the current
     * output will retain its value but will be overwritten with the new current value in the next
     * iteration. */
    for (int i = FILTER_ORDER; i >= 1; i--) {
      outputs[i] = outputs[i - 1];
    }
  }
}
