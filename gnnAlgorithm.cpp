/**
 * @file gnnAlgorithm.cpp
 * @brief This is the main file for the GCNN algorithm. It reads in the command line arguments and then runs the GCNN algorithm.
 * The GCNN currently uses the following algorithm: 
 * Performs the following convolutions on the graph:
 * 
 * 1. Generate the node embeddings for the graph.
 * 2. Create the neural network and apply it to the node embeddings.
 * 3. Backpropagate the neural network to update the weights and biases.
 * 4. Forward propogate the neural network to get the accuracy and loss.
 * 5. Repeat steps 2-4 for the number of epochs given by the user.
 * 6. Forward propogate the neural network to get the accuracy and loss for the test data.
 * 7. Print the results.
 * 
 * The current neural network architecture is the following:
 * 100 -> 47 -> Leaky ReLU Activation Function -> 47 -> Softmax Activation Function
 * 
 * The algorithm uses the mean square error as the loss function and the accuracy function.
 * We specify the convolution function as averaging the node embeddings of the neighbours of each node to neighbours one hop from each node.
 * We then perform the aggregation function on that updated node embedding to get the new node embedding on the two hop neighbours.
 * We can perform this aggregation function for as many convolutions as we want.
 * 
 * 
 * <cuDNN IMPLEMENTATION OF THE CODE>
 * 
 * @note This code is not performance portable on other platforms. It is designed to run on NVIDIA GPUs for maximum performance.
 * I would recommend setting the following environment variables to get the best performance:
 * export OMP_NUM_THREADS=<insert number of threads>
 * export OMP_PROC_BIND=true
 * export OMP_PLACES=threads
 * 
 */

//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology  Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

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

#include "checkSizes.h"
#include "getGraphSize.hpp"
#include "readGraphData.hpp"
#include "initialiseWeights.hpp"
#include "accuracyErrorUtil.hpp"
#include "generateNodeEmbeddings.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>


#ifdef USING_SILO
  #include <silo.h>
  #include "writeTimestep.h"
#endif

#ifdef USING_THRUST
  #include <thrust/sort.h>
  #include <thrust/execution_policy.h>
#endif

#include <Kokkos_Core.hpp>
#include <Kokkos_NestedSort.hpp>
#include <Kokkos_StaticCrsGraph.hpp>

__global__ void fillMatrix(float* vector, float* matrix, int train_size, int test_size, int hidden_layer_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < test_size * hidden_layer_size) {
        int vectorIdx = idx / test_size;
        matrix[idx] = vector[vectorIdx];
    }
}

__global__ void updateMatrix(float* vector, float* matrix, int m, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < m * n) {
        int vectorIdx = idx / m;
        matrix[idx] += vector[vectorIdx];
    }
}

__global__ void fillOnes(float* vector, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < size) {
        vector[idx] = 1.0f;
    }
}

__global__ void leakyRelu(float* vector, float* out, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < size) {
    out[idx] = vector[idx];
    if (out[idx] <= 0.0f) {
      out[idx] *= 0.2f;
    } 
  }
}

__global__ void deltaLeakyRelu(float* vector, float* out, int size) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx < size) {
    if (vector[idx] < 0.0f) {
      out[idx] *= 0.2f;
    }
  }
}

/**
 * @brief The following program is the starting point for the GNN algorithm. It reads in the command line arguments and then runs the GNN algorithm.
 * The current neural network architecture is a 1 input layer, 1 hidden layer and 1 output layer. The input layer has the same number of neurons as the number of features in the graph.
 * The hidden layer has a user defined number of neurons and the output layer has the same number of neurons as the number of classes in the graph (47).
 * 
 * Hence the neural network architecture is as follows:
 * 100 -> 47 -> Leaky ReLU Activation Function -> 47 Softmax Activation Function
 * 
 * The algorithm uses the mean square error as the loss function and the accuracy as the accuracy function.
 * The algorithm uses the backpropagation algorithm to update the weights and biases of the neural network.
 * 
 * <THIS IS THE cuDNN VERSION>
 * 
 * WARNING: This code is not performance portable on other platforms. It is designed to run on NVIDIA GPUs for maximum performance.
 * Utilises the cuDNN library for the neural network operations and cuBLAS for the matrix operations.
 * 
 * @param Train 
 * @param Test
 * @param Hidden
 * @param Epochs
 * @param Convolutions
 * @param Learning_Rate
 * @param Accuracy_Threshold
 * 
 * @return int 
 */
int main( int argc, char* argv[] )
{
  long long train_size = -1;
  long long test_size = -1;
  long long hidden_layer_size = -1; // nice
  long long num_epochs = -1;
  long long convolutions = -1;
  float learning_rate = -0.0f;
  float accuracy_threshold = -0.0f;

  std::chrono::duration<double> initialisationTime;
  std::chrono::duration<double> generateNodeEmbeddingTime;
  std::chrono::duration<double> neuralNetworkTime;
  std::chrono::duration<double> backpropagateTime;
  std::chrono::duration<double> forwardpropagateTime;
  std::chrono::duration<double> utilityTime;

  // Read command line arguments.
  for ( int i = 0; i < argc; i++ ) {
    if ( ( strcmp( argv[ i ], "-Train" ) == 0 ) || ( strcmp( argv[ i ], "-Train Size" ) == 0 ) ) {
      train_size = atoi( argv[ ++i ] );
      printf( "  User Train Size is %ld\n", train_size );
    }
    if ( ( strcmp( argv[ i ], "-Test" ) == 0 ) || ( strcmp( argv[ i ], "-Test Size" ) == 0 ) ) {
      test_size = atoi( argv[ ++i ] );
      printf( "  User Test Size is %ld\n", test_size );
    }
    if ( ( strcmp( argv[ i ], "-Hidden" ) == 0 ) || ( strcmp( argv[ i ], "-Hidden Layer Size" ) == 0 ) ) {
      hidden_layer_size = atoi( argv[ ++i ] );
      printf( "  User Hidden Layer Size is %ld\n", hidden_layer_size );
    }
    if ( ( strcmp( argv[ i ], "-E" ) == 0 ) || ( strcmp( argv[ i ], "-Epochs" ) == 0 ) ) {
      num_epochs = atoi( argv[ ++i ] );
      printf( "  User Number of Epochs is %ld\n", num_epochs );
    }
    if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-Convolutions" ) == 0 ) ) {
      convolutions = atoi( argv[ ++i ] );
      printf( "  User Number of Convolutions is %ld\n", convolutions );
    }
    if ( ( strcmp( argv[ i ], "-LR" ) == 0 ) || ( strcmp( argv[ i ], "-Learning Rate" ) == 0 ) ) {
      learning_rate = atof( argv[ ++i ] );
      printf( "  User Learning Rate is %f\n", learning_rate );
    }
    if ( ( strcmp( argv[ i ], "-AT" ) == 0 ) || ( strcmp( argv[ i ], "-Accuracy Threshold" ) == 0 ) ) {
      accuracy_threshold = atof( argv[ ++i ] );
      printf( "  User Accuracy Threshold is %f\n", accuracy_threshold );
    }


    else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
      fprintf(stdout,  "  GNN Options:\n" );
      fprintf(stdout,  "  -Train Size        (-Train )  <long>:   num, determines number of training data points (num) (default: 800)\n" );
      fprintf(stdout,  "  -Test Size         (-Test )   <long>:   num, determines number of testing data points (num) (default: 200)\n" );
      fprintf(stdout,  "  -Hidden Layer Size (-Hidden ) <long>:   num, determines number of neurons in the hidden layer (num) (default: 47)\n" );
      fprintf(stdout,  "  -Epochs            (-E )      <long>:   num, determines the number of epochs (num) (default: 100)\n" );
      fprintf(stdout,  "  -Convolutions      (-C )      <long>:   num, determines the number of convolutions (num) (default: 1)\n" );
      fprintf(stdout,  "  -Learning Rate     (-LR )     <float>:  num, determines the learning rate (num) (default: 0.1)\n" );
      fprintf(stdout,  "  -Accuracy Threshold(-AT )     <float>:  num, determines the accuracy threshold (num) (default: 100.1)\n" );
      fprintf(stdout,  "  -help              (-h ):         print this message\n\n" );
      exit( 1 );
    }
  }

  //Error check the sizes given by the user
  checkSizes(train_size, test_size, hidden_layer_size, num_epochs, convolutions, learning_rate, accuracy_threshold);

  learning_rate = learning_rate / (float)train_size;
  learning_rate = -1.0 * learning_rate;

  long long num_edges;
  long long num_vertices;
  long long feature_size;

  Kokkos::initialize( argc, argv );
  {

  int numThreads;
  if (getenv("OMP_NUM_THREADS"))
    numThreads = std::atoi(getenv("OMP_NUM_THREADS"));
  fprintf(stdout, "\n================================================================================\n");
  fprintf(stdout, "Number of Threads is : %d.\n", numThreads);
  fprintf(stdout, "\n================================================================================\n");


  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    #define MemSpace Kokkos::CudaSpace
    #define MemLayout Kokkos::LayoutRight
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    fprintf(stdout, "<ONLY APPLICABLE TO AMPERE ADA GPUS>  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           128,
           128 *
               deviceProp.multiProcessorCount);
  #endif
  #if !defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    #define MemSpace Kokkos::OpenMP
    #define MemLayout Kokkos::LayoutRight
    fprintf(stderr, "OpenMP is not supported on this implementation. Exiting...\n");
    exit(1);
  #endif
  #ifdef KOKKOS_ENABLE_HIP // (if we want to add support for Radeon GPUs later)
    #define MemSpace Kokkos::Experimental::HIPSpace
    #define MemLayout Kokkos::LayoutRight
    fprintf(stderr, "HIP is not supported on this implementation. Exiting...\n");
    exit(1);
  #endif
  #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #define MemLayout Kokkos::LayoutRight
    fprintf(stderr, "HostSpace is not supported on this implementation. Exiting...\n");
    exit(1);
  #endif

  using ExecutionSpace = MemSpace::execution_space;
  using StaticCrsGraphType = Kokkos::StaticCrsGraph<int, Kokkos::LayoutRight, MemSpace, Kokkos::MemoryTraits<Kokkos::RandomAccess>, int>;

  //Start timers for the algorithm
  auto start = std::chrono::high_resolution_clock::now();

  std::string vertex_size_file = std::string("../DEEPgnnAlgorithm/products/raw/num-node-list.csv");
  std::string edge_size_file = std::string("../DEEPgnnAlgorithm/products/raw/num-edge-list.csv");
  std::string edge_file = std::string("../DEEPgnnAlgorithm/products/raw/edge.csv");
  std::string feature_file = std::string("../DEEPgnnAlgorithm/products/raw/node-feat.csv");
  std::string label_file = std::string("../DEEPgnnAlgorithm/products/raw/node-label.csv");


  //Read in the graph size data from the files
  if (getGraphSize(vertex_size_file, edge_size_file, feature_file, &num_vertices, &num_edges, &feature_size) != 0) {
    fprintf(stdout, "Failed to read in the graph size data.\n");
    return -1;
  }

  // For the graph structure
  std::vector<std::vector<int>> graph_edges(num_vertices * 2);
  Kokkos::View<float**, MemLayout, MemSpace> node_features("node_features", num_vertices, feature_size);
  Kokkos::View<float**, MemLayout, MemSpace> node_embeddings_layer_1("node_embeddings_layer_1", num_vertices, feature_size);
  Kokkos::View<float**, MemLayout, MemSpace> node_embeddings_layer_2("node_embeddings_layer_2", num_vertices, feature_size);
  Kokkos::View<int*, MemSpace> node_labels("node_labels", num_vertices);

  Kokkos::View<float**, MemLayout, MemSpace> expected_results("expected_results", num_vertices, 47);

  fprintf(stdout, "<Number of vertices: %d>\n", num_vertices);
  fprintf(stdout, "<Number of edges: %d>\n", num_edges);
  fprintf(stdout, "<Feature size: %d>\n", feature_size);

  //readGraphData(label_file, feature_file, edge_file, node_labels, node_features, &graph_edges, num_vertices, num_edges, feature_size);
  //InitialiseExpected(expected_results, node_labels, num_vertices);

  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_node_features = Kokkos::create_mirror_view(node_features);
    Kokkos::View<int*, MemSpace>::HostMirror h_node_labels = Kokkos::create_mirror_view(node_labels);
    Kokkos::View<float**, MemLayout, MemSpace>::HostMirror h_expected_results = Kokkos::create_mirror_view(expected_results);
  #endif

  // Initialise Cuda memory spaces for the graph data and performs copies between the host device and the GPU device
  #if defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)
    // Read in the graph data from the files
    readGraphData(label_file, feature_file, edge_file, h_node_labels, h_node_features, &graph_edges, num_vertices, num_edges, feature_size);
    InitialiseExpected(h_expected_results, h_node_labels, num_vertices, hidden_layer_size);
    Kokkos::deep_copy(node_labels, h_node_labels);
    Kokkos::deep_copy(node_features, h_node_features);
    Kokkos::deep_copy(expected_results, h_expected_results);
  #endif
  #if (!defined(KOKKOS_ENABLE_CUDA) && defined(KOKKOS_ENABLE_OPENMP)) || defined(KOKKOS_ENABLE_SERIAL)
    readGraphData(label_file, feature_file, edge_file, node_labels, node_features, &graph_edges, num_vertices, num_edges, feature_size);
    InitialiseExpected(expected_results, node_labels, num_vertices, hidden_layer_size);
  #endif

  StaticCrsGraphType d_graph;
  StaticCrsGraphType::HostMirror h_graph;

  d_graph = Kokkos::create_staticcrsgraph<StaticCrsGraphType>("d_graph", graph_edges);
  h_graph = Kokkos::create_mirror(d_graph);

  auto end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the function
  initialisationTime += end - start;
  
  start = std::chrono::high_resolution_clock::now();

  // Create the first level of the graph
  for (long long i = 0; i < convolutions; i++) {
    if (i == 0) {
      GenerateNodeEmbedding(d_graph, node_features, node_embeddings_layer_1, 0, feature_size, num_vertices);
      node_features = node_embeddings_layer_1;
    }
    else {
      if (i % 2 == 0) {
        GenerateNodeEmbedding(d_graph, node_embeddings_layer_2, node_embeddings_layer_1, 0, feature_size, num_vertices);
        node_features = node_embeddings_layer_1;
      }
      else {
        GenerateNodeEmbedding(d_graph, node_embeddings_layer_1, node_embeddings_layer_2, 0, feature_size, num_vertices);
        node_features = node_embeddings_layer_2;
      }
    }
  }

  Kokkos::fence();

  end = std::chrono::high_resolution_clock::now();

  //Store the time it took to run the function
  generateNodeEmbeddingTime += end - start;

  start = std::chrono::high_resolution_clock::now();

  int numGPUs;
  cudaGetDeviceCount(&numGPUs);
  fprintf(stdout, "<Found: %d GPUs>.\n", numGPUs);
  cudaSetDevice(0); // use GPU0
  int device; 
  struct cudaDeviceProp devProp;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&devProp, device);
  fprintf(stdout, "<Compute capability: %d.%d>\n", devProp.major, devProp.minor);

  // Create the neural network using the cudnn library
  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  fprintf(stdout, "<Created cuDNN handle>\n");

  // create the tensor descriptor
  cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int n = train_size, c = 1, h = 1, w = 100;
  int NUM_ELEMENTS = n*c*h*w;
  int NUM_ELEMENTS_TEST = test_size*c*h*w;
  if (!NUM_ELEMENTS_TEST) NUM_ELEMENTS_TEST = 1;
  cudnnTensorDescriptor_t x_desc;
  cudnnTensorDescriptor_t test_desc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnCreateTensorDescriptor(&test_desc);
  cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, hidden_layer_size, h, c);
  cudnnSetTensor4dDescriptor(test_desc, format, dtype, test_size, hidden_layer_size, h, c);

  // create the tensor
  float *x_host = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));

  float *weights_host_1, *biases_host_1, *output_host_1, *transposed_output_host_1;
  weights_host_1 = (float *)std::malloc(hidden_layer_size * w * sizeof(float));
  biases_host_1 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));
  output_host_1 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));
  transposed_output_host_1 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));

  float *weights_host_2, *biases_host_2, *output_host_2, *transposed_output_host_2;
  weights_host_2 = (float *)std::malloc(hidden_layer_size * hidden_layer_size * sizeof(float));
  biases_host_2 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));
  output_host_2 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));
  transposed_output_host_2 = (float *)std::malloc(NUM_ELEMENTS * sizeof(float));

  float *output_2_test_host;
  output_2_test_host = (float *)std::malloc(NUM_ELEMENTS_TEST * sizeof(float));

  InitialiseWeights(weights_host_1, weights_host_2, feature_size, hidden_layer_size, hidden_layer_size);

  float *x, *x_transpose, *weights_1, *biases_1, *output_1, *transposed_output_1;

  cudaMalloc(&x, NUM_ELEMENTS * sizeof(float));
  cudaMalloc(&x_transpose, NUM_ELEMENTS * sizeof(float));
  cudaMalloc(&weights_1, hidden_layer_size * w * sizeof(float));
  cudaMalloc(&biases_1, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&output_1, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&transposed_output_1, n*c*h*hidden_layer_size * sizeof(float));

  cudaMemcpy(x, node_features.data(), NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToDevice);  
  cudaMemcpy(weights_1, weights_host_1, hidden_layer_size * w * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(biases_1, biases_host_1, n*c*h*hidden_layer_size  * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_1, output_host_1, n*c*h*hidden_layer_size * sizeof(float), cudaMemcpyHostToDevice);

  float *weights_2, *biases_2, *output_2, *transposed_output_2;

  cudaMalloc(&weights_2, hidden_layer_size * hidden_layer_size * sizeof(float));
  cudaMalloc(&biases_2, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&output_2, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&transposed_output_2, n*c*h*hidden_layer_size * sizeof(float));

  cudaMemcpy(weights_2, weights_host_2, hidden_layer_size * hidden_layer_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(biases_2, biases_host_2, n*c*h*hidden_layer_size  * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(output_2, output_host_2, n*c*h*hidden_layer_size * sizeof(float), cudaMemcpyHostToDevice);

  float * expected_results_cuda = expected_results.data();

  // allocate memory for gradients layers
  float *gradients_1, *gradients_2,  *gradients_1_transpose, *gradients_2_transpose;
  cudaMalloc(&gradients_1, n*c*h*w * sizeof(float));
  cudaMalloc(&gradients_2, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&gradients_2_transpose, n*c*h*hidden_layer_size * sizeof(float));
  cudaMalloc(&gradients_1_transpose, n*c*h*w * sizeof(float));

  float *bias_store_1, *bias_store_2;
  cudaMalloc(&bias_store_1, hidden_layer_size * n * sizeof(float));
  cudaMalloc(&bias_store_2, hidden_layer_size * n * sizeof(float));

  float * ones, *bias_update_1, *bias_update_2;
  cudaMalloc(&ones, n * sizeof(float));
  cudaMalloc(&bias_update_1, hidden_layer_size * sizeof(float));
  cudaMalloc(&bias_update_2, hidden_layer_size * sizeof(float));

  float *test_set, *biases_1_test, *biases_2_test, *output_1_test, *output_2_test, *transposed_output_1_test, *transposed_output_2_test;
  cudaMalloc(&test_set, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&biases_1_test, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&biases_2_test, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&output_1_test, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&output_2_test, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&transposed_output_1_test, NUM_ELEMENTS_TEST * sizeof(float));
  cudaMalloc(&transposed_output_2_test, NUM_ELEMENTS_TEST * sizeof(float));

  cudaMemcpy(test_set, node_features.data() + (train_size + 0) * w, NUM_ELEMENTS_TEST * sizeof(float), cudaMemcpyDeviceToDevice);

  // apply relu activation function on the biases matrix
  // using cudnn activation forward
  cudnnActivationDescriptor_t activation_desc;
  cudnnCreateActivationDescriptor(&activation_desc);
  cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0f);

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t output_desc_test;
  cudnnCreateTensorDescriptor(&output_desc);
  cudnnCreateTensorDescriptor(&output_desc_test);
  cudnnSetTensor4dDescriptor(output_desc, format, dtype, n, hidden_layer_size, h, c);
  cudnnSetTensor4dDescriptor(output_desc_test, format, dtype, test_size, hidden_layer_size, h, c);

  // performs normalisation of input data before being passed to the softmax function to avoid formulation of NAN values
  cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE;
  cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL;


  // perform the forward propogation x = x*w + b
  // utilise GEMM 
  float alpha = 1.0f;
  float beta = 1.0f;
  float alpha_transpose = 1.0f;
  float beta_transpose = 0.0f;
  float alpha_activation = 1.0f;
  float beta_activation = 0.0f;
  float alpha_neg_one = -1.0f;
  float final_correct = 0.0f;
  cublasHandle_t handle;
  cublasCreate(&handle);

  fprintf(stdout, "||========================================================RESULTS========================================================\n");

  end = std::chrono::high_resolution_clock::now();
  initialisationTime += end - start;

  int blockSize = 256;
  int gridSize1 = (n * hidden_layer_size + blockSize - 1) / blockSize;
  int gridSize2 = (n + blockSize - 1) / blockSize;
  int gridSize3 = (test_size * hidden_layer_size + blockSize - 1) / blockSize;

  //transpose x
  cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, w, n, &alpha_transpose, x, n, &beta_transpose, nullptr, w, x_transpose, w);

  fillOnes<<<gridSize2, blockSize>>>(ones, n);  

  for (int i = 0; i < num_epochs; i++) {

    cudaDeviceSynchronize();
    Kokkos::fence();

    start = std::chrono::high_resolution_clock::now(); // actual start of the neural network

    auto forward_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(bias_store_1, biases_1, hidden_layer_size * n * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      n, hidden_layer_size, w,
                                      &alpha,
                                      x, CUDA_R_32F, w,
                                      weights_1, CUDA_R_32F, w,
                                      &beta,
                                      biases_1, CUDA_R_32F, n,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // apply relu activation function on the biases matrix
    //cudnnActivationForward(cudnn, activation_desc, &alpha_activation, x_desc, biases_1, &beta_activation, x_desc, output_1);
    leakyRelu<<<gridSize1, blockSize>>>(biases_1, output_1, hidden_layer_size * n);

    // Transpose the output matrix
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     output_1, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     transposed_output_1, hidden_layer_size);

    // the x matrix is 3xhidden_layer_size, the weights matrix is hidden_layer_sizexhidden_layer_size, the output matrix is 3xhidden_layer_size
    cudaMemcpy(bias_store_2, biases_2, hidden_layer_size * n * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      n, hidden_layer_size, hidden_layer_size,
                                      &alpha,
                                      transposed_output_1, CUDA_R_32F, hidden_layer_size,
                                      weights_2, CUDA_R_32F, hidden_layer_size,
                                      &beta,
                                      biases_2, CUDA_R_32F, n,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // transpose the output matrix
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     biases_2, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     transposed_output_2, hidden_layer_size);

    // Now perform softmax normalisation on the output matrix
    // using cudnn softmax forward
    // we dont want to apply softmax on the entire matrix, just in strides of hidden_layer_size
    cudnnSoftmaxForward(cudnn, algo, mode, &alpha_activation, output_desc, transposed_output_2, &beta_activation, output_desc, output_2);

    cudaDeviceSynchronize();
    Kokkos::fence();

    auto forward_end = std::chrono::high_resolution_clock::now();

    forwardpropagateTime += forward_end - forward_start;

    /*===============================================================================================================================================
    |FORWARD PROPOGATION DONE|                                                                                            |BACKWARD PROPOGATION START|
    ===============================================================================================================================================*/

    auto backpropagate_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(gradients_2, expected_results_cuda, n*c*h*hidden_layer_size * sizeof(float), cudaMemcpyDeviceToDevice);

    //multiply the gradients_2 matrix by -1
    cublasSscal(handle, n*c*h*hidden_layer_size, &alpha_neg_one, gradients_2, 1);
    // find difference between output and expected
    cublasSaxpy(handle, n*c*h*hidden_layer_size, &alpha, output_2, 1, gradients_2, 1);

    // compute gradients for gradients_1 using the gradients_2
    // in matrix notation we perform (gradients_2 * weights_2)
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                      n, hidden_layer_size, hidden_layer_size,
                                      &alpha,
                                      gradients_2, CUDA_R_32F, hidden_layer_size,
                                      weights_2, CUDA_R_32F, hidden_layer_size,
                                      &beta_transpose,
                                      gradients_1, CUDA_R_32F, n,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // transpose gradients_2 and gradients_1
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     gradients_2, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     gradients_2_transpose, hidden_layer_size);

    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     gradients_1, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     gradients_1_transpose, hidden_layer_size);

    // gradients_2 is nxhidden_layer_size, transposed_output_1 is hidden_layer_sizexn, hence matrix multiply should output hidden_layer_sizexhidden_layer_size
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                      hidden_layer_size, hidden_layer_size, n,
                                      &learning_rate,
                                      transposed_output_1, CUDA_R_32F, hidden_layer_size,
                                      gradients_2, CUDA_R_32F, hidden_layer_size,
                                      &alpha,
                                      weights_2, CUDA_R_32F, hidden_layer_size,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    deltaLeakyRelu<<<gridSize1, blockSize>>>(transposed_output_1, gradients_1_transpose, hidden_layer_size * n);

    // update weights_1
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                      w, hidden_layer_size, n,
                                      &learning_rate,
                                      x, CUDA_R_32F, w,
                                      gradients_1_transpose, CUDA_R_32F, hidden_layer_size,
                                      &alpha,
                                      weights_1, CUDA_R_32F, w,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // update biases_1
    cudaMemcpy(biases_1, bias_store_1, hidden_layer_size * n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(biases_2, bias_store_2, hidden_layer_size * n * sizeof(float), cudaMemcpyDeviceToDevice);

    // cublasGemmEx on gradients_2_transpose and ones
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                      1, hidden_layer_size, n,
                                      &learning_rate,
                                      ones, CUDA_R_32F, 1,
                                      gradients_2, CUDA_R_32F, hidden_layer_size,
                                      &beta_transpose,
                                      bias_update_2, CUDA_R_32F, 1,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // same for biases_1
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                      1, hidden_layer_size, n,
                                      &learning_rate,
                                      ones, CUDA_R_32F, 1,
                                      gradients_1_transpose, CUDA_R_32F, hidden_layer_size,  // this was a fluke, with enough epochs it will eventually become unstable if you transpose gradients_1
                                      &beta_transpose,
                                      bias_update_1, CUDA_R_32F, 1,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    updateMatrix<<<gridSize2, blockSize>>>(bias_update_2, biases_2, n, hidden_layer_size);

    cudaDeviceSynchronize();
    Kokkos::fence();

    updateMatrix<<<gridSize1, blockSize>>>(bias_update_1, biases_1, n, hidden_layer_size);      

    cudaDeviceSynchronize();
    Kokkos::fence();

    auto backpropagate_end = std::chrono::high_resolution_clock::now();

    end = std::chrono::high_resolution_clock::now();

    backpropagateTime += backpropagate_end - backpropagate_start;

    neuralNetworkTime += end - start;

    /*================================================================================================================================================
    |BACKWARD PROPOGATION DONE|                                                                                            |ACCURACY CALCULATION START|
    ================================================================================================================================================*/

    auto utility_start = std::chrono::high_resolution_clock::now();

    float accuracy = CalculateAccuracy(output_2, node_labels, n, 0, hidden_layer_size);
    final_correct = accuracy;
    float loss = MeanSquareError(output_2, expected_results, n, hidden_layer_size);  
    fprintf(stdout, "|| Epoch: %d, Loss: %f, Accuracy: %f%\n", i, loss, accuracy);

    cudaDeviceSynchronize();
    Kokkos::fence();

    auto utility_end = std::chrono::high_resolution_clock::now();

    utilityTime += utility_end - utility_start;
    if (accuracy > accuracy_threshold) {
      i = num_epochs;
    }
  }

  Kokkos::fence();

  if (train_size != num_vertices) {

    start = std::chrono::high_resolution_clock::now();
    // transpose biases_1 and biases_2 into biases_1_store and biases_2_store
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     biases_1, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     bias_store_1, hidden_layer_size);
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, n,
                                     &alpha_transpose,
                                     biases_2, n,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     bias_store_2, hidden_layer_size);


    fillMatrix<<<gridSize3, blockSize>>>(bias_store_1, biases_1_test, train_size, test_size, hidden_layer_size);
    fillMatrix<<<gridSize3, blockSize>>>(bias_store_2, biases_2_test, train_size, test_size, hidden_layer_size);

    cudaDeviceSynchronize();

    // perform forward propogation on the train set and calculate accuracy
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      (int)test_size, (int)hidden_layer_size, w,
                                      &alpha,
                                      test_set, CUDA_R_32F, w,
                                      weights_1, CUDA_R_32F, w,
                                      &beta,
                                      biases_1_test, CUDA_R_32F, (int)test_size,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

                                      

    // apply relu activation function on the biases matrix
    //cudnnActivationForward(cudnn, activation_desc, &alpha_activation, test_desc, biases_1_test, &beta_activation, test_desc, output_1_test);
    leakyRelu<<<gridSize3, blockSize>>>(biases_1_test, output_1_test, hidden_layer_size * test_size);
    // Transpose the output matrix
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, test_size,
                                     &alpha_transpose,
                                     output_1_test, test_size,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     transposed_output_1_test, hidden_layer_size);
    // the x matrix is 3xhidden_layer_size, the weights matrix is hidden_layer_sizexhidden_layer_size, the output matrix is 3xhidden_layer_size

    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      test_size, hidden_layer_size, hidden_layer_size,
                                      &alpha,
                                      transposed_output_1_test, CUDA_R_32F, hidden_layer_size,
                                      weights_2, CUDA_R_32F, hidden_layer_size,
                                      &beta,
                                      biases_2_test, CUDA_R_32F, test_size,
                                      CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    // transpose the output matrix
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     hidden_layer_size, test_size,
                                     &alpha_transpose,
                                     biases_2_test, test_size,
                                     &beta_transpose,
                                     nullptr, hidden_layer_size,
                                     transposed_output_2_test, hidden_layer_size);

    // Now perform softmax normalisation on the output matrix
    // using cudnn softmax forward
    // we dont want to apply softmax on the entire matrix, just in strides of hidden_layer_size
    cudnnSoftmaxForward(cudnn, algo, mode, &alpha_activation, output_desc_test, transposed_output_2_test, &beta_activation, output_desc_test, output_2_test);

    cudaDeviceSynchronize();

    end = std::chrono::high_resolution_clock::now();
    neuralNetworkTime += end - start;
    // this is a forward propogation on the test set time
    forwardpropagateTime += end - start;

    // perform accuracy calculation
    auto utility_start = std::chrono::high_resolution_clock::now();

    cudaMemcpy(output_2_test_host, output_2_test, NUM_ELEMENTS_TEST * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    final_correct = CalculateAccuracy(output_2_test, node_labels, test_size, train_size + 0, hidden_layer_size);

    Kokkos::fence();

    auto utility_end = std::chrono::high_resolution_clock::now();

    utilityTime += utility_end - utility_start;
  }

  cublasDestroy(handle);

  Kokkos::fence();

  // Destroy the cudnn handle
  cudnnDestroy(cudnn);
  Kokkos::fence();
  cudaFree(x);
  cudaFree(weights_1);
  cudaFree(biases_1);
  cudaFree(output_1);
  cudaFree(transposed_output_1);

  cudaFree(weights_2);
  cudaFree(biases_2);
  cudaFree(output_2);
  cudaFree(transposed_output_2);

  cudaFree(gradients_1);
  cudaFree(gradients_2);
  cudaFree(gradients_2_transpose);
  cudaFree(gradients_1_transpose);

  cudaFree(bias_store_1);
  cudaFree(bias_store_2);

  cudaFree(ones);
  cudaFree(bias_update_1);
  cudaFree(bias_update_2);

  cudaFree(test_set);
  cudaFree(biases_1_test);
  cudaFree(biases_2_test);
  cudaFree(output_1_test);
  cudaFree(output_2_test);
  cudaFree(transposed_output_1_test);
  cudaFree(transposed_output_2_test);

  std::free(x_host);
  std::free(weights_host_1);
  std::free(biases_host_1);
  std::free(output_host_1);
  std::free(transposed_output_host_1);

  std::free(weights_host_2);
  std::free(biases_host_2);
  std::free(output_host_2);
  std::free(transposed_output_host_2);

  std::free(output_2_test_host);
  
  auto totalTime = initialisationTime + generateNodeEmbeddingTime + neuralNetworkTime + utilityTime;

  fprintf(stdout, "||========================================================RESULTS========================================================\n");

  fprintf(stdout, "\n||============================RESULTS============================\n");
  fprintf(stdout, "||Initialisation Time:                                %.6f   \n", initialisationTime.count());
  fprintf(stdout, "||Generate Node Embedding Time:                       %.6f   \n", generateNodeEmbeddingTime.count());
  fprintf(stdout, "||Neural Network Time:                                %.6f   \n", neuralNetworkTime.count());
  fprintf(stdout, "||Forward Propagate Time:                             %.6f   \n", forwardpropagateTime.count());
  fprintf(stdout, "||Backward Propagate Time:                            %.6f   \n", backpropagateTime.count());
  fprintf(stdout, "||Utility Time:                                       %.6f   \n", utilityTime.count());
  fprintf(stdout, "||Total Time:                                         %.6f   \n", totalTime.count());
  fprintf(stdout, "||============================RESULTS============================\n");
  fprintf(stdout, "||Test Accuracy:                                      %.6f%\n", final_correct);
  fprintf(stdout, "||============================RESULTS============================\n");

  }
  Kokkos::finalize();
  
  return 0;
}

