#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <random>
#include <vector>
#include <utility>
#include <stdlib.h>

#include "accuracyErrorUtil.hpp"

/**
 * @brief Calculate the mean square error loss.
 * 
 * @param forward_propogate_results_layer2 The results of the forward propagation in layer 2.
 * @param expected_results The expected results for each vertex.
 * @param num_vertices The total number of vertices.
 * @param neurons_per_layer The number of neurons per layer.
 * @return float The mean square error loss.
 */
float MeanSquareError(ForwardPropogateResultsType forward_propogate_results_layer2,
                       ExpectedResultsType expected_results,
                       int num_vertices,
                       int neurons_per_layer) {

  // Calculate the mean square error loss
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;
  float loss = 0.0f;
  Kokkos::parallel_reduce( "CalcMSE", current_range_policy( 0, num_vertices ), KOKKOS_LAMBDA ( int i, float &update ) {
    float diff;
    int index = i*neurons_per_layer;
    for (int j = 0; j < 47; ++j) {
      diff = expected_results(i, j) - forward_propogate_results_layer2[index + j];
      update += diff * diff;
    }
  }, loss );

  Kokkos::fence();

  loss /= (float)num_vertices;
  loss *= 0.5f;

  return loss;
}

/**
 * @brief CalculateAccuracy calculates the accuracy of the neural network.
 * 
 * @param forward_propogate_results_layer2 The results of the forward propagation in layer 2.
 * @param node_labels The labels of the nodes.
 * @param num_vertices The total number of vertices.
 * @param start_index The starting index for calculation.
 * @param neurons_per_layer The number of neurons per layer.
 * @return float The accuracy of the neural network.
 */
float CalculateAccuracy( ForwardPropogateResultsType forward_propogate_results_layer2,
                         NodeLabelsTypeDevice node_labels,
                         int num_vertices,
                         int start_index,
                         int neurons_per_layer) {

  // Calculate the accuracy of the neural network
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Static>>;
  int total = num_vertices;
  int correct = 0;
  Kokkos::parallel_reduce( "CalcAcc", current_range_policy( start_index, start_index + num_vertices ), KOKKOS_LAMBDA ( int i, int &update ) {
    float curr_max = 0.0f;
    int max_index = 0;
    int index = (i - start_index)*neurons_per_layer;
    for (int j = 0; j < neurons_per_layer; ++j) {
      if (forward_propogate_results_layer2[index + j] > curr_max) {
        curr_max = forward_propogate_results_layer2[index + j];
        max_index = j;
      }
    } 
    if (max_index == node_labels(i)) {
      update += 1;
    }
  }, correct );

  Kokkos::fence();

  return ((float)correct/(float)total)*100.0f;
}