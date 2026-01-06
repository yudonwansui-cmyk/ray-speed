/**
 * @file kernel.cuh
 * @brief 声明将在GPU上执行的CUDA Kernel函数。
 *
 * 该文件定义了与仿真核心计算相关的GPU Kernel的接口。
 * 这些Kernel由 `main.cu` 中的主机代码启动，并在 `kernel.cu` 中实现。
 */

#ifndef KERNEL_CUH
#define KERNEL_CUH

 // 包含CUDA随机数生成器的内核API
#include <curand_kernel.h>
// 包含我们自定义的仿真数据结构，如Heliostat, Receiver, AtomicCounters等
#include "simulation.cuh"

// ==================================================================================
// Kernel 声明
// ==================================================================================

/**
 * @brief 执行核心的双向蒙特卡洛光线追踪仿真。
 *
 * 每个GPU线程负责一个微面元（定日镜表面的一个离散小单元）。
 * 对于每个微面元，线程将：
 * 1. 模拟镜面法线的微观扰动。
 * 2. 在一个基于物理模型的太阳光锥内采样多条光线。
 * 3. 对每条采样光线，执行阴影（向太阳）和遮挡/截断（向吸收塔）的碰撞检测。
 * 4. 使用3D-DDA网格加速碰撞检测。
 * 5. 原子地更新全局计数器以统计结果。
 *
 * @param d_heliostats              [in] 指向存储所有定日镜数据的GPU数组的指针。
 * @param num_heliostats            [in] 定日镜的总数。
 * @param d_receiver                [in] 指向存储吸收塔数据的GPU结构体的指针。
 * @param d_grid                    [in] 指向存储3D-DDA加速网格元数据的GPU结构体的指针。
 * @param sun_direction             [in] 当前时间点下，指向太阳的单位方向向量。
 * @param d_rand_states             [in/out] 指向已初始化的curandState数组的指针，Kernel将使用并更新这些状态。
 * @param d_counters                [out] 指向用于原子计数的GPU结构体的指针，用于累加阴影、遮挡和击中光线数。
 * @param rays_per_microfacet       [in] 每个微面元需要追踪的光线数量（光锥采样密度）。
 * @param microfacets_per_heliostat_x [in] 单个定日镜在宽度方向上的微面元数量。
 * @param sun_cone_half_angle_rad   [in] 太阳光锥的半角，单位为弧度。
 * @param normal_perturbation_sigma_rad [in] 镜面法线扰动的标准差（高斯分布），单位为弧度。
 */
__global__ void rayTracingKernel(
    const Heliostat* d_heliostats,
    int num_heliostats,
    const Receiver* d_receiver,
    const AccelerationGrid* d_grid,
    float3 sun_direction,
    HeliostatStats* d_stats,
    int rays_per_microfacet,
    int microfacets_per_heliostat_x,
    float sun_cone_half_angle_rad,
    float normal_perturbation_sigma_rad
);


#endif // KERNEL_CUH