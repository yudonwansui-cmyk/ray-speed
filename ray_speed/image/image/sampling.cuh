/**
 * @file sampling.cuh
 * @brief 提供在GPU上执行的各种随机采样函数。
 *
 * 该文件包含了为蒙特卡洛光线追踪提供随机变量的核心函数，包括：
 * 1. 模拟镜面粗糙度的微面元法线扰动采样。
 * 2. 模拟太阳有限尺寸的太阳光锥内光线方向采样。
 * 所有函数均为 __device__ inline，并依赖于CURAND库来生成随机数。
 */

#ifndef SAMPLING_CUH
#define SAMPLING_CUH

#include <curand_kernel.h>
#include "vector_math.cuh"

 // ==================================================================================
 // 随机采样辅助函数
 // ==================================================================================

 /**
  * @brief 使用Box-Muller变换从两个均匀分布的随机数生成一个标准正态分布的随机数。
  * @param state [in/out] 指向当前线程的CURAND状态的指针。
  * @return 一个服从 N(0, 1) 分布的随机浮点数。
  */
__device__ inline float sampleNormalDistribution(curandState* state) {
    // curand_uniform会生成 (0, 1] 范围的随机数，我们需要避免log(0)
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}


// ==================================================================================
// 核心采样函数
// ==================================================================================

/**
 * @brief 对理想法线进行随机扰动，以模拟镜面的微观不平整（斜率误差）。[已修正版本]
 *
 * 扰动被建模为一个二维圆形高斯分布。这意味着反射光束的能量分布也是
 * 一个二维圆形高斯分布，其标准差为 2 * sigma_rad。
 * 算法步骤：
 * 1. 在 [0, 2*PI] 均匀采样方位角 phi。
 * 2. 通过对均匀分布变量应用逆变换采样，生成一个服从瑞利分布的极角 theta。
 *    瑞利分布的标准差为 sigma_rad。
 * 3. 将球坐标(theta, phi)下的扰动向量转换到世界坐标系并应用。
 *
 * @param ideal_normal              [in] 宏观镜面的理想单位法向量。
 * @param state                     [in/out] 指向当前线程的CURAND状态的指针。
 * @param sigma_rad                 [in] 镜面斜率误差的标准差，单位为弧度。
 * @return 扰动后的单位法向量。
 */
__device__ inline float3 sampleNormalPerturbation(const float3& ideal_normal, curandState* state, float sigma_rad) {
    // 1. 采样一个均匀分布的方位角 phi
    float phi = 2.0f * M_PI * curand_uniform(state);

    // 2. 使用逆变换采样，从均匀分布生成一个服从瑞利分布的极角 theta
    //    瑞利分布的CDF是 F(r) = 1 - exp(-r^2 / (2*sigma^2))
    //    令 u = F(r)，其中 u 是(0,1)的均匀随机数，反解出 r (即theta)
    //    theta = sigma * sqrt(-2 * log(u))
    float u = curand_uniform(state);
    float theta = sigma_rad * sqrtf(-2.0f * logf(u));

    // 3. 将扰动(theta, phi)转换为局部坐标系下的向量
    //    z' = cos(theta)
    //    x' = sin(theta) * cos(phi)
    //    y' = sin(theta) * sin(phi)
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);
    float3 perturbation = make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cos_theta);

    // 4. 创建一个以ideal_normal为Z轴的局部坐标系 (与之前相同)
    float3 T, B;
    if (abs(ideal_normal.x) > abs(ideal_normal.y)) {
        float invLen = 1.0f / sqrtf(ideal_normal.x * ideal_normal.x + ideal_normal.z * ideal_normal.z);
        T = make_float3(ideal_normal.z * invLen, 0.0f, -ideal_normal.x * invLen);
    }
    else {
        float invLen = 1.0f / sqrtf(ideal_normal.y * ideal_normal.y + ideal_normal.z * ideal_normal.z);
        T = make_float3(0.0f, ideal_normal.z * invLen, -ideal_normal.y * invLen);
    }
    B = cross(ideal_normal, T);

    // 5. 将局部坐标系中的扰动向量转换到世界坐标系
    float3 perturbed_normal = perturbation.x * T + perturbation.y * B + perturbation.z * ideal_normal;

    return perturbed_normal; // 这个法线现在是物理正确的
}


/**
 * @brief 在一个以sun_direction为轴的锥体内均匀采样一个光线方向。
 *
 * 这是对太阳光锥的简化模型。更精确的模型如Buie模型会考虑盘面亮度不均。
 * 算法通过在锥底的圆盘上均匀采样一点，然后将该点与锥顶点（观察者）相连得到方向。
 *
 * @param sun_direction         [in] 指向太阳中心的单位方向向量。
 * @param state                 [in/out] 指向当前线程的CURAND状态的指针。
 * @param half_angle_rad        [in] 光锥的半角，单位为弧度。
 * @return 光锥内一个随机采样的单位方向向量。
 */
__device__ inline float3 sampleSunConeUniform(const float3& sun_direction, curandState* state, float half_angle_rad) {
    // 1. 计算光锥的最大cos值
    float cos_theta_max = cosf(half_angle_rad);

    // 2. 在 [cos_theta_max, 1] 范围内均匀采样一个cos值
    float cos_theta = cos_theta_max + curand_uniform(state) * (1.0f - cos_theta_max);

    // 3. 计算对应的sin值
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

    // 4. 在 [0, 2*PI] 范围内均匀采样一个方位角 phi
    float phi = 2.0f * M_PI * curand_uniform(state);

    // 5. 将球坐标 (sin_theta, cos_theta, phi) 转换为局部笛卡尔坐标
    float3 local_dir = make_float3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);

    // 6. 创建一个以sun_direction为Z轴的局部坐标系
    float3 T, B;
    if (abs(sun_direction.x) > abs(sun_direction.y)) {
        float invLen = 1.0f / sqrtf(sun_direction.x * sun_direction.x + sun_direction.z * sun_direction.z);
        T = make_float3(sun_direction.z * invLen, 0.0f, -sun_direction.x * invLen);
    }
    else {
        float invLen = 1.0f / sqrtf(sun_direction.y * sun_direction.y + sun_direction.z * sun_direction.z);
        T = make_float3(0.0f, sun_direction.z * invLen, -sun_direction.y * invLen);
    }
    B = cross(sun_direction, T);

    // 7. 将局部方向向量转换到世界坐标系
    float3 world_dir = local_dir.x * T + local_dir.y * B + local_dir.z * sun_direction;

    return world_dir;
}


/**
 * @brief [占位符] 在太阳光锥内根据Buie模型采样一个光线方向。
 *
 * Buie模型是一个更精确的太阳辐照度分布模型，它考虑了太阳盘面的"边缘变暗"效应。
 * 它的实现比均匀采样更复杂，通常涉及到逆变换采样或拒绝采样。
 *
 * @note 为了简化，本项目默认使用 sampleSunConeUniform。如果需要更高精度，
 *       可以替换为此函数的完整实现。
 *
 * @param sun_direction         [in] 指向太阳中心的单位方向向量。
 * @param state                 [in/out] 指向当前线程的CURAND状态的指针。
 * @param half_angle_rad        [in] 光锥的半角，单位为弧度。
 * @return 光锥内一个根据Buie模型采样的单位方向向量。
 */
__device__ inline float3 sampleSunConeBuie(const float3& sun_direction, curandState* state, float half_angle_rad) {
    // Buie模型的实现较为复杂，此处为了项目能直接运行，
    // 我们暂时调用均匀采样模型作为替代。
    // 在实际的高精度研究中，应在此处实现完整的Buie模型。
    return sampleSunConeUniform(sun_direction, state, half_angle_rad);
}


#endif // SAMPLING_CUH