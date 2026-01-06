/**
 * @file vector_math.cuh
 * @brief 一个轻量级的、基于CUDA float3的3D向量数学库。
 *
 * 提供了在主机(CPU)和设备(GPU)代码中均可使用的3D向量操作。
 * 所有函数均为内联函数以最大化性能，并重载了标准运算符以便于使用。
 * 设计目标是简洁、高效和数值稳定。
 */

#ifndef VECTOR_MATH_CUH
#define VECTOR_MATH_CUH

#include <cuda_runtime.h>
#include <cmath> // For sqrtf, etc.

 // ==================================================================================
 // 常量定义
 // ==================================================================================
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ==================================================================================
// 运算符重载 (Operator Overloads)
// ==================================================================================

// 向量加法
__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline void operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// 向量减法
__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__host__ __device__ inline float3 operator-(const float3& a) { // 负号运算符
    return make_float3(-a.x, -a.y, -a.z);
}

// 向量与标量乘法
__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline float3 operator*(float s, const float3& a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline void operator*=(float3& a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
}

// 向量与标量除法
__host__ __device__ inline float3 operator/(const float3& a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}
__host__ __device__ inline void operator/=(float3& a, float s) {
    float inv = 1.0f / s;
    a.x *= inv; a.y *= inv; a.z *= inv;
}

// ==================================================================================
// 核心向量函数 (Core Vector Functions)
// ==================================================================================

/**
 * @brief 计算两个向量的点积 (Dot Product)。
 */
__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * @brief 计算两个向量的叉积 (Cross Product)。
 */
__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

/**
 * @brief 计算向量的模长 (Length or Magnitude)。
 */
__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

/**
 * @brief 计算向量的模长平方 (Squared Length)。在仅比较长度时，此函数比length()更高效。
 */
__host__ __device__ inline float length_sq(const float3& v) {
    return dot(v, v);
}

/**
 * @brief 将向量归一化为单位向量 (Normalize)。
 * @param v 要归一化的向量。
 * @return 单位向量。如果输入为零向量，则返回零向量以避免除零错误。
 */
__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    if (len > 1e-8f) { // 使用epsilon防止除以零
        return v / len;
    }
    return make_float3(0.0f, 0.0f, 0.0f);
}

/**
 * @brief 计算入射向量关于法线的反射向量 (Reflection)。[已根据张平论文Eq(14)最终修正]
 * @param i 入射单位向量 (从光源指向表面)。
 * @param n 表面的单位法向量。
 * @return 反射单位向量。
 */
__host__ __device__ inline float3 reflect(const float3& i, const float3& n) {
    // 严格遵循参考论文《太阳能塔式光热镜场光学效率计算方法》中的 Equation (14):
    // V'_r = 2 * cos_theta * V_n - V_s  (其中 V_s 是入射向量 i, V_n 是法线 n)
    return i - 2.0f * dot(i, n) * n;
}

/**
 * @brief 线性插值 (Linear Interpolation)。
 * @param a 起始向量。
 * @param b 结束向量。
 * @param t 插值因子 (0.0 to 1.0)。
 * @return 插值结果向量。
 */
__host__ __device__ inline float3 lerp(const float3& a, const float3& b, float t) {
    return a + t * (b - a);
}

/**
 * @brief 打印向量到控制台 (仅主机代码可用)。
 */
inline void print(const float3& v, const char* name) {
#if !defined(__CUDA_ARCH__) // 仅在主机代码中编译
    printf("%s: (%.4f, %.4f, %.4f)\n", name, v.x, v.y, v.z);
#endif
}


#endif // VECTOR_MATH_CUH