#pragma once

#include "vector.hpp"

Vector vectorAdd(const Vector& vec1, const Vector& vec2);
void vectorInplaceAdd(Vector& vec1, const Vector& vec2);

Vector vectorSub(const Vector& vec1, const Vector& vec2);
float vectorNorm(const Vector& vec, const float p);