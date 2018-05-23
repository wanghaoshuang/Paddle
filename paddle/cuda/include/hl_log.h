#ifndef HL_LOG_H
#define HL_LOG_H
#pragma once
#include <iostream>
#include "cuda_runtime.h"
using namespace std;
template<typename T>
void log_cuda_data(T* cudaData, int size) {
  T* cpuData = (T*)malloc(size * sizeof(T));
  cudaMemcpy(cpuData, cudaData, size * sizeof(T), cudaMemcpyDeviceToHost);
  cout << endl;
  for (int i=0; i<size; ++i) {
    cout << cpuData[i] << ",";
  }
  cout << endl;
  free(cpuData);
}

template<typename T>
void log_cuda_data(const T* cudaData, int size) {
  T* cpuData = (T*)malloc(size * sizeof(T));
  cudaMemcpy(cpuData, cudaData, size * sizeof(T), cudaMemcpyDeviceToHost);
  cout << endl;
  for (int i=0; i<size; ++i) {
    cout << cpuData[i] << ",";
  }
  cout << endl;
  free(cpuData);
}


#endif // HL_LOG_H
