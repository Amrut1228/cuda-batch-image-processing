
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t error = (call);                                              \
    if (error != cudaSuccess) {                                              \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)               \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';            \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

constexpr int kDefaultSignals = 512;
constexpr int kDefaultSamples = 4096;
constexpr int kDefaultRadius = 2;
constexpr int kThreadsPerBlock = 256;

__global__ void smoothSignals(const float* input,
                              float* smoothed,
                              int numSignals,
                              int samplesPerSignal,
                              int radius) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSamples = numSignals * samplesPerSignal;

  if (globalIndex >= totalSamples) {
    return;
  }

  int sampleIndex = globalIndex % samplesPerSignal;

  float sum = 0.0f;
  int count = 0;

  int start = max(0, sampleIndex - radius);
  int end = min(samplesPerSignal - 1, sampleIndex + radius);

  for (int i = start; i <= end; ++i) {
    sum += input[globalIndex - sampleIndex + i];
    ++count;
  }

  smoothed[globalIndex] = sum / static_cast<float>(count);
}

__global__ void computeMaxAbs(const float* data,
                              float* maxAbs,
                              int numSignals,
                              int samplesPerSignal) {
  int signal = blockIdx.x;

  if (signal >= numSignals) {
    return;
  }

  __shared__ float sharedMax[kThreadsPerBlock];

  float localMax = 0.0f;
  int base = signal * samplesPerSignal;

  for (int i = threadIdx.x; i < samplesPerSignal; i += blockDim.x) {
    localMax = fmaxf(localMax, fabsf(data[base + i]));
  }

  sharedMax[threadIdx.x] = localMax;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      sharedMax[threadIdx.x] =
          fmaxf(sharedMax[threadIdx.x],
                sharedMax[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    maxAbs[signal] = sharedMax[0];
  }
}

__global__ void normalizeSignals(const float* smoothed,
                                  const float* maxAbs,
                                  float* output,
                                  int numSignals,
                                  int samplesPerSignal) {
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int totalSamples = numSignals * samplesPerSignal;

  if (globalIndex >= totalSamples) {
    return;
  }

  int signal = globalIndex / samplesPerSignal;
  float scale = fmaxf(maxAbs[signal], 1.0e-6f);

  output[globalIndex] =
      fmaxf(-1.0f, fminf(1.0f, smoothed[globalIndex] / scale));
}

void generateSignals(std::vector<float>& data,
                     int numSignals,
                     int samplesPerSignal) {
  std::mt19937 generator(42);
  std::normal_distribution<float> noise(0.0f, 0.20f);

  for (int signal = 0; signal < numSignals; ++signal) {
    float frequency1 = 2.0f + 0.15f * static_cast<float>(signal % 10);
    float frequency2 = 7.0f + 0.20f * static_cast<float>(signal % 7);

    for (int sample = 0; sample < samplesPerSignal; ++sample) {
      float t = static_cast<float>(sample) / samplesPerSignal;

      float clean =
          0.70f * std::sin(2.0f * static_cast<float>(M_PI) *
                           frequency1 * t) +
          0.25f * std::cos(2.0f * static_cast<float>(M_PI) *
                           frequency2 * t);

      data[signal * samplesPerSignal + sample] = clean + noise(generator);
    }
  }
}

void processSignalsCPU(const std::vector<float>& input,
                       std::vector<float>& output,
                       int numSignals,
                       int samplesPerSignal,
                       int radius) {
  std::vector<float> smoothed(input.size());
  std::vector<float> maxAbs(numSignals, 0.0f);

  for (int signal = 0; signal < numSignals; ++signal) {
    int base = signal * samplesPerSignal;

    for (int sample = 0; sample < samplesPerSignal; ++sample) {
      float sum = 0.0f;
      int count = 0;

      int start = std::max(0, sample - radius);
      int end = std::min(samplesPerSignal - 1, sample + radius);

      for (int i = start; i <= end; ++i) {
        sum += input[base + i];
        ++count;
      }

      smoothed[base + sample] =
          sum / static_cast<float>(count);

      maxAbs[signal] =
          std::max(maxAbs[signal], std::fabs(smoothed[base + sample]));
    }
  }

  for (int signal = 0; signal < numSignals; ++signal) {
    int base = signal * samplesPerSignal;
    float scale = std::max(maxAbs[signal], 1.0e-6f);

    for (int sample = 0; sample < samplesPerSignal; ++sample) {
      float value = smoothed[base + sample] / scale;
      output[base + sample] =
          std::max(-1.0f, std::min(1.0f, value));
    }
  }
}

void writeSampleOutput(const std::vector<float>& data,
                       int samplesPerSignal) {
  std::ofstream file("sample_processed.csv");

  if (!file.is_open()) {
    std::cerr << "Warning: could not create sample_processed.csv\n";
    return;
  }

  file << "signal_id,sample_id,value\n";
  int samplesToWrite = std::min(32, samplesPerSignal);

  for (int sample = 0; sample < samplesToWrite; ++sample) {
    file << 0 << "," << sample << ","
         << std::fixed << std::setprecision(6)
         << data[sample] << '\n';
  }
}

void parseArguments(int argc,
                    char** argv,
                    int& numSignals,
                    int& samplesPerSignal,
                    int& radius) {
  for (int i = 1; i < argc; ++i) {
    std::string argument = argv[i];

    if (argument == "--signals" && i + 1 < argc) {
      numSignals = std::stoi(argv[++i]);
    } else if (argument == "--samples" && i + 1 < argc) {
      samplesPerSignal = std::stoi(argv[++i]);
    } else if (argument == "--radius" && i + 1 < argc) {
      radius = std::stoi(argv[++i]);
    } else if (argument == "--help") {
      std::cout
          << "Usage: ./signal_batch [options]\n"
          << "  --signals N   Number of signals (default 512)\n"
          << "  --samples N   Samples per signal (default 4096)\n"
          << "  --radius N    Moving-average radius (default 2)\n";
      std::exit(EXIT_SUCCESS);
    } else {
      std::cerr << "Unknown argument: " << argument << '\n';
      std::exit(EXIT_FAILURE);
    }
  }

  if (numSignals <= 0 || samplesPerSignal <= 0 || radius < 0) {
    std::cerr << "Arguments must be positive; radius cannot be negative.\n";
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  int numSignals = kDefaultSignals;
  int samplesPerSignal = kDefaultSamples;
  int radius = kDefaultRadius;

  parseArguments(argc, argv, numSignals, samplesPerSignal, radius);

  int totalSamples = numSignals * samplesPerSignal;
  size_t dataBytes = static_cast<size_t>(totalSamples) * sizeof(float);
  size_t maxAbsBytes = static_cast<size_t>(numSignals) * sizeof(float);

  std::cout << "CUDA Batch Signal Processing\n";
  std::cout << "Signals: " << numSignals << '\n';
  std::cout << "Samples per signal: " << samplesPerSignal << '\n';
  std::cout << "Total samples: " << totalSamples << '\n';
  std::cout << "Moving-average radius: " << radius << '\n';

  cudaDeviceProp device{};
  CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

  std::cout << "GPU: " << device.name << '\n';
  std::cout << "Global memory: "
            << static_cast<double>(device.totalGlobalMem) /
                   (1024.0 * 1024.0 * 1024.0)
            << " GB\n";

  std::vector<float> input(totalSamples);
  std::vector<float> cpuOutput(totalSamples);
  std::vector<float> gpuOutput(totalSamples);

  generateSignals(input, numSignals, samplesPerSignal);

  auto cpuStart = std::chrono::high_resolution_clock::now();
  processSignalsCPU(input,
                    cpuOutput,
                    numSignals,
                    samplesPerSignal,
                    radius);
  auto cpuEnd = std::chrono::high_resolution_clock::now();

  double cpuMilliseconds =
      std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();

  float* dInput = nullptr;
  float* dSmoothed = nullptr;
  float* dMaxAbs = nullptr;
  float* dOutput = nullptr;

  CUDA_CHECK(cudaMalloc(&dInput, dataBytes));
  CUDA_CHECK(cudaMalloc(&dSmoothed, dataBytes));
  CUDA_CHECK(cudaMalloc(&dMaxAbs, maxAbsBytes));
  CUDA_CHECK(cudaMalloc(&dOutput, dataBytes));

  auto gpuTotalStart = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(dInput,
                        input.data(),
                        dataBytes,
                        cudaMemcpyHostToDevice));

  int totalBlocks =
      (totalSamples + kThreadsPerBlock - 1) / kThreadsPerBlock;

  smoothSignals<<<totalBlocks, kThreadsPerBlock>>>(
      dInput,
      dSmoothed,
      numSignals,
      samplesPerSignal,
      radius);

  CUDA_CHECK(cudaGetLastError());

  computeMaxAbs<<<numSignals, kThreadsPerBlock>>>(
      dSmoothed,
      dMaxAbs,
      numSignals,
      samplesPerSignal);

  CUDA_CHECK(cudaGetLastError());

  normalizeSignals<<<totalBlocks, kThreadsPerBlock>>>(
      dSmoothed,
      dMaxAbs,
      dOutput,
      numSignals,
      samplesPerSignal);

  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(gpuOutput.data(),
                        dOutput,
                        dataBytes,
                        cudaMemcpyDeviceToHost));

  auto gpuTotalEnd = std::chrono::high_resolution_clock::now();

  double gpuTotalMilliseconds =
      std::chrono::duration<double, std::milli>(
          gpuTotalEnd - gpuTotalStart).count();

  cudaEvent_t startEvent;
  cudaEvent_t stopEvent;

  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  CUDA_CHECK(cudaEventRecord(startEvent));

  smoothSignals<<<totalBlocks, kThreadsPerBlock>>>(
      dInput,
      dSmoothed,
      numSignals,
      samplesPerSignal,
      radius);

  computeMaxAbs<<<numSignals, kThreadsPerBlock>>>(
      dSmoothed,
      dMaxAbs,
      numSignals,
      samplesPerSignal);

  normalizeSignals<<<totalBlocks, kThreadsPerBlock>>>(
      dSmoothed,
      dMaxAbs,
      dOutput,
      numSignals,
      samplesPerSignal);

  CUDA_CHECK(cudaEventRecord(stopEvent));
  CUDA_CHECK(cudaEventSynchronize(stopEvent));

  float kernelMilliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernelMilliseconds,
                                  startEvent,
                                  stopEvent));

  CUDA_CHECK(cudaFree(dInput));
  CUDA_CHECK(cudaFree(dSmoothed));
  CUDA_CHECK(cudaFree(dMaxAbs));
  CUDA_CHECK(cudaFree(dOutput));
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));

  float maxError = 0.0f;

  for (size_t i = 0; i < gpuOutput.size(); ++i) {
    maxError =
        std::max(maxError, std::fabs(cpuOutput[i] - gpuOutput[i]));
  }

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "CPU processing time: "
            << cpuMilliseconds << " ms\n";
  std::cout << "GPU end-to-end time: "
            << gpuTotalMilliseconds << " ms\n";
  std::cout << "GPU kernel time: "
            << kernelMilliseconds << " ms\n";
  std::cout << "CPU/GPU max absolute error: "
            << maxError << '\n';

  if (maxError < 1.0e-4f) {
    std::cout << "Validation: PASS\n";
  } else {
    std::cout << "Validation: FAIL\n";
  }

  std::cout << "Throughput: "
            << (static_cast<double>(totalSamples) /
                (gpuTotalMilliseconds * 1000.0))
            << " million samples/second\n";

  writeSampleOutput(gpuOutput, samplesPerSignal);

  CUDA_CHECK(cudaDeviceReset());

  return (maxError < 1.0e-4f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
