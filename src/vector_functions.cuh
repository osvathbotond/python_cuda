#pragma once


void addCuda(const float* d_vec1, const float* d_vec2, float* d_res, const size_t vector_length);
void subCuda(const float* d_vec1, const float* d_vec2, float* d_res, const size_t vector_length);
void scaleCuda(const float* d_vec, const float scalar, float* d_res, const size_t vector_length);
float normCuda(const float* d_vec, const float p, const size_t vector_length);
void sinCuda(const float* d_vec, float* d_res, const size_t vector_length);
