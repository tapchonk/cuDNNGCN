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

#include "generateNodeEmbeddings.hpp"

/**
 * @brief Generate the node embedding by utilising a mean aggregation of the node features and the node neighbourhood
 * We can stack the operations for each layer of the graph, if we for example wanted to extend the convolutions over
 * a two neighbourhood layer, we would just need to perform this function twice and so on. 
 * 
/**
 * @param graph: The graph representation containing the connectivity information between nodes.
 * @param node_features: The matrix containing the features of each node.
 * @param node_embeddings_layer_1: The matrix to store the generated node embeddings for the first layer.
 * @param num_vertices: The total number of vertices in the graph.
 * @param feature_size: The size of the feature vector for each node.
 * @param node_embed: The number of nodes to generate embeddings for.
 * @return int: Returns 1 upon successful generation of node embeddings.
 */
int GenerateNodeEmbedding(KokkosGraphType graph,
                          KokkosNodeType node_features,
                          KokkosNodeEmbeddingType node_embeddings_layer_1,
                          long long start_index,
                          long long feature_size,
                          long long node_embed) {

  // Generate the node embedding by utilising a mean aggregation
  // Each layer of neighbourhoods will be aggregated into the node embedding with the average of that nodes layer out degree
  // We also include the node features from the starting node in the node embedding
  using current_range_policy = Kokkos::RangePolicy<MemSpace::execution_space, Kokkos::Schedule<Kokkos::Dynamic>>;
  
  Kokkos::parallel_for(Kokkos::TeamPolicy<MemSpace::execution_space>(node_embed, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<MemSpace::execution_space>::member_type& team1) {
      int rank = team1.league_rank();
      float normalised_length_1 = Kokkos::sqrt((float)(graph.rowConst(rank).length + 1));
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team1, feature_size), [=] (int feature) {
        node_features(rank, feature) = node_features(rank, feature) / normalised_length_1;
      });
  });

  Kokkos::parallel_for(Kokkos::TeamPolicy<MemSpace::execution_space>(node_embed, Kokkos::AUTO),
    KOKKOS_LAMBDA (const Kokkos::TeamPolicy<MemSpace::execution_space>::member_type& team1) {
      
      int rank = team1.league_rank();
      auto row_view_level_1 = graph.rowConst(rank);
      int length_1 = row_view_level_1.length + 1;
      if (length_1 == 1) {
        // if the node has no neighbours, just use the node features
        for (int feature = 0; feature < feature_size; ++feature) {
          node_embeddings_layer_1(rank, feature) = node_features(rank, feature);
        }
        return;
      }
      float length_1_div = (float)length_1;

      // for each of the neighbours:
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team1, feature_size), [=] (int feature) {
        float sum = 0.0f;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team1, length_1 - 1), [=] (int j, float& temp_sum) {
          temp_sum += node_features(row_view_level_1(j), feature);
        }, sum);
        node_embeddings_layer_1(rank, feature) = sum;
        // also take into account the source node features
        node_embeddings_layer_1(rank, feature) += node_features(rank, feature);
        // all nodes have equal weights
        // Divide the node embedding by the number of nodes in the neighbourhood
        node_embeddings_layer_1(rank, feature) /= Kokkos::sqrt((float)length_1_div);
      });

      return;

  });

  return 1;
}

