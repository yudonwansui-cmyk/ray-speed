/**
 * @file kernel.cu
 * @brief 实现将在GPU上执行的CUDA Kernel函数。[最终版]
 *
 * 此文件包含了仿真核心计算逻辑的实现，包括：
 * 1. rayTracingKernel: 主光线追踪Kernel，每个线程处理一个微面元。
 *    [修改] 内核现在为每个定日镜独立统计光线数据，而非一个全局总数。
 */

#include "kernel.cuh"
#include "intersections.cuh" // 引入几何求交函数
#include "sampling.cuh"      // 引入随机采样函数
#include <device_launch_parameters.h>
#include "cuda_runtime.h"

 // ==================================================================================
 // Kernel 实现: rayTracingKernel
 // ==================================================================================
__global__ void rayTracingKernel(
    const Heliostat* d_heliostats,
    int num_heliostats,
    const Receiver* d_receiver,
    const AccelerationGrid* d_grid,
    float3 sun_direction,
    // [修改] 参数从 AtomicCounters* 变为 HeliostatStats*
    HeliostatStats* d_stats,
    int rays_per_microfacet,
    int microfacets_per_heliostat_x,
    float sun_cone_half_angle_rad,
    float normal_perturbation_sigma_rad
) {
    // --- 1. 线程身份识别 ---
    // 计算当前线程负责的全局微面元索引
    unsigned long long global_microfacet_idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // 计算每个定日镜上的微面元总数
    const int microfacets_per_heliostat = microfacets_per_heliostat_x * microfacets_per_heliostat_x;
    if (global_microfacet_idx >= (unsigned long long)num_heliostats * microfacets_per_heliostat) {
        return; // 超出范围的线程直接退出
    }

    // 计算所属的定日镜ID和在该定日镜上的局部ID
    int heliostat_id = global_microfacet_idx / microfacets_per_heliostat;
    int local_microfacet_idx = global_microfacet_idx % microfacets_per_heliostat;

    // 从局部ID计算出在定日镜表面上的二维索引(i, j)
    int j = local_microfacet_idx / microfacets_per_heliostat_x;
    int i = local_microfacet_idx % microfacets_per_heliostat_x;

    // --- 2. 加载数据到快速内存 (寄存器) ---
    const Heliostat h = d_heliostats[heliostat_id];
    curandState local_rand_state;
    // 使用一个固定的种子和全局唯一的线程ID来初始化，保证可复现性
    curand_init(12345ULL, global_microfacet_idx, 0, &local_rand_state);

    // --- 3. 微面元建模 ---
    // a. 计算微面元在定日镜局部坐标系下的中心位置
    float microfacet_size = h.width / microfacets_per_heliostat_x;
    float local_x_pos = (i + 0.5f) * microfacet_size - h.width * 0.5f;
    float local_y_pos = (j + 0.5f) * microfacet_size - h.height * 0.5f;

    // b. 将局部位置转换到世界坐标系 (使用与求交函数一致的坐标系构建方法)
    float3 up_vector = make_float3(0.0f, 0.0f, 1.0f);
    float3 local_x_axis, local_y_axis;
    if (abs(dot(h.ideal_normal, up_vector)) > 0.9999f) {
        local_x_axis = make_float3(1.0f, 0.0f, 0.0f);
        local_y_axis = normalize(cross(h.ideal_normal, local_x_axis));
    }
    else {
        local_y_axis = normalize(cross(h.ideal_normal, up_vector));
        local_x_axis = normalize(cross(local_y_axis, h.ideal_normal));
    }
    float3 microfacet_world_pos = h.center + local_x_pos * local_x_axis + local_y_pos * local_y_axis;

    // c. 对理想法线进行随机扰动，模拟镜面不平整
    float3 perturbed_normal = sampleNormalPerturbation(h.ideal_normal, &local_rand_state, normal_perturbation_sigma_rad);

    // --- 4. 光锥追踪循环 ---
    for (int k = 0; k < rays_per_microfacet; ++k) {
        // a. 在太阳光锥内采样一条入射光线
        float3 incident_dir = sampleSunConeBuie(sun_direction, &local_rand_state, sun_cone_half_angle_rad);

        // b. 计算反射光线方向 (根据扰动后的真实法线)
        float3 reflect_dir = reflect(-incident_dir, perturbed_normal);

        // --- c. 阴影测试 (向太阳方向追) ---
        Ray shadow_ray;
        shadow_ray.origin = microfacet_world_pos;
        shadow_ray.direction = -incident_dir;
        shadow_ray.origin = shadow_ray.origin + shadow_ray.direction * 1e-4f; // 偏移起点避免自相交

        if (traverseGridAndIntersect(shadow_ray, heliostat_id, *d_grid, d_heliostats)) {
            // [修改] 使用heliostat_id作为索引，更新对应定日镜的阴影计数器
            atomicAdd(&d_stats[heliostat_id].shadow_rays, 1);
            continue; // 被阴影遮挡，追踪下一条光线
        }

        // --- d. 遮挡测试 (向吸收塔方向追) ---
        Ray block_ray;
        block_ray.origin = microfacet_world_pos;
        block_ray.direction = reflect_dir;
        block_ray.origin = block_ray.origin + block_ray.direction * 1e-3f; // 偏移起点避免自相交

        if (traverseGridAndIntersect(block_ray, heliostat_id, *d_grid, d_heliostats)) {
            // [修改] 使用heliostat_id作为索引，更新对应定日镜的遮挡计数器
            atomicAdd(&d_stats[heliostat_id].blocked_rays, 1);
            continue; // 被其他镜子遮挡，追踪下一条光线
        }

        // --- e. 截断测试 (与吸收塔求交) ---
        float t_intersection;
        if (intersectRayCylinder(block_ray, *d_receiver, t_intersection)) {
            if (t_intersection > 0) {
                // [修改] 使用heliostat_id作为索引，更新对应定日镜的击中计数器
                atomicAdd(&d_stats[heliostat_id].hit_rays, 1);
            }
        }
    }
}