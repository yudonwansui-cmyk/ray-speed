/**
 * @file intersections.cuh
 * @brief 提供在GPU上执行的高性能几何求交测试函数。
 *
 * 包含光线与基本几何体（平面、圆柱体）的求交函数，以及
 * 用于加速场景遍历的3D-DDA（数字微分分析器）网格遍历算法。
 * 所有函数均为 __device__ inline 函数，专为在CUDA Kernel中调用而设计。
 */

#ifndef INTERSECTIONS_CUH
#define INTERSECTIONS_CUH

#include "vector_math.cuh"
#include "simulation.cuh"
#include <algorithm>

 // 定义一个小的epsilon值用于浮点数比较，以处理精度问题
#define INTERSECTION_EPSILON 1e-5f

// ==================================================================================
// 核心数据结构
// ==================================================================================

/**
 * @struct Ray
 * @brief 定义一条光线，包含起点和单位方向向量。
 */
struct Ray {
    float3 origin;
    float3 direction;
};


// ==================================================================================
// 基本几何体求交函数
// ==================================================================================

/**
 * @brief 测试光线与一个有界平面（矩形定日镜）的交点。[已修正版本]
 * @param ray           [in]  待测试的光线。
 * @param h             [in]  定日镜对象，定义了平面的位置、法线和边界。
 * @param t_out         [out] 如果相交，返回光线起点到交点的距离t。
 * @return true 如果光线在矩形边界内与平面相交，否则返回false。
 */
__device__ inline bool intersectRayPlane(const Ray& ray, const Heliostat& h, float& t_out) {
    const float3 plane_normal = h.ideal_normal;
    float denominator = dot(ray.direction, plane_normal);

    if (abs(denominator) < INTERSECTION_EPSILON) {
        return false;
    }

    float3 origin_to_center = h.center - ray.origin;
    float t = dot(origin_to_center, plane_normal) / denominator;

    if (t < INTERSECTION_EPSILON) {
        return false;
    }

    // --- [修正] 检查交点是否在矩形边界内 ---
    float3 intersection_point = ray.origin + t * ray.direction;
    float3 vec_to_intersection = intersection_point - h.center;

    // --- [修正] 使用更稳健的方法建立局部坐标系以避免数值不稳定 ---
// 目标仍然是创建一个局部X轴是水平的坐标系 (z=0)。
    float3 up_vector = make_float3(0.0f, 0.0f, 1.0f);
    float3 local_x_axis, local_y_axis;

    // 如果法线非常接近垂直于地面(与up_vector平行)，
    // 叉乘会不稳定。在这种特殊情况下，我们将X轴定义为世界X轴。
    if (abs(dot(h.ideal_normal, up_vector)) > 0.9999f) {
        local_x_axis = make_float3(1.0f, 0.0f, 0.0f);
        local_y_axis = normalize(cross(h.ideal_normal, local_x_axis));
    }
    else {
        // ***** 这是关键的修改 *****
     // 确保这里的定义顺序和 intersections.cuh 中的定义完全一样
        local_y_axis = normalize(cross(h.ideal_normal, up_vector));
        local_x_axis = normalize(cross(local_y_axis, h.ideal_normal));
    }

    // *** [关键补充] 将被删除的投影计算加回来 ***
  // 将交点到中心的向量投影到局部X和Y轴上
    float local_x = dot(vec_to_intersection, local_x_axis);
    float local_y = dot(vec_to_intersection, local_y_axis);

    // 检查是否超出边界
    if (abs(local_x) > h.width * 0.5f || abs(local_y) > h.height * 0.5f) {
        return false;
    }

    // 检查是否超出边界
    if (abs(local_x) > h.width * 0.5f || abs(local_y) > h.height * 0.5f) {
        return false;
    }

    t_out = t;
    return true;
}


/**
 * @brief 测试光线与一个Z轴对齐的有限高度圆柱体（吸收塔）的交点。
 * @param ray           [in]  待测试的光线。
 * @param r             [in]  吸收塔对象。
 * @param t_out         [out] 如果相交，返回光线起点到交点的距离t。
 * @return true 如果光线与圆柱体表面相交，否则返回false。
 */
__device__ inline bool intersectRayCylinder(const Ray& ray, const Receiver& r, float& t_out) {
    // 将光线转换到以圆柱体底面中心为原点的坐标系
    float3 oc = ray.origin - r.center;

    // 构造求解交点距离t的二次方程 At^2 + Bt + C = 0
    // A = dx^2 + dy^2
    float a = ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y;



    // --- [最终修正] 增加数值稳定性检查 ---
    // 如果 'a' 非常小, 意味着光线几乎平行于圆柱体的Z轴。
    // 这种光线无法与圆柱体的侧面相交，直接判为未命中，以避免除零错误。
    if (a < INTERSECTION_EPSILON) { // 使用和 intersectRayPlane 相同的 epsilon
        return false;
    }



    // B = 2 * (ox*dx + oy*dy)
    float b = 2.0f * (oc.x * ray.direction.x + oc.y * ray.direction.y);
    // C = ox^2 + oy^2 - R^2
    float c = oc.x * oc.x + oc.y * oc.y - r.radius * r.radius;

    // 求解判别式
    float discriminant = b * b - 4.0f * a * c;

    // 如果判别式<0，光线与无限长圆柱体不相交
    if (discriminant < 0) {
        return false;
    }

    float sqrt_discriminant = sqrtf(discriminant);
    float inv_2a = 1.0f / (2.0f * a);
    float t0 = (-b - sqrt_discriminant) * inv_2a;
    float t1 = (-b + sqrt_discriminant) * inv_2a;

    float t = -1.0f;
    // 我们需要第一个在光线前进方向上的交点 (t > epsilon)
    if (t0 > INTERSECTION_EPSILON) {
        t = t0;
    }
    else if (t1 > INTERSECTION_EPSILON) {
        t = t1;
    }
    else {
        // 两个交点都在后面
        return false;
    }

    // 计算交点的z坐标
    float intersection_z = ray.origin.z + t * ray.direction.z;

    // 检查z坐标是否在圆柱体的高度范围内
    if (intersection_z >= r.center.z && intersection_z <= r.center.z + r.height) {
        t_out = t;
        return true;
    }

    // 如果第一个交点在高度范围外，可能光线从顶/底盖穿入，
    // 但题目模型是受光外表面，所以我们不考虑顶盖和底盖的交点。
    return false;
}


// ==================================================================================
// 3D-DDA 网格遍历与求交
// ==================================================================================

/**
 * @brief 使用3D-DDA算法遍历加速网格，并测试与格子内的定日镜的碰撞。[最终修正版]
 */
__device__ inline bool traverseGridAndIntersect(const Ray& ray, int originating_h_id, const AccelerationGrid& grid, const Heliostat* d_heliostats) {
    // --- 1. DDA参数初始化 ---
    int3 cell_idx;
    int3 step;
    float3 t_max;
    float3 t_delta;

    float3 start_pos = ray.origin - grid.world_min;

    // --- X轴初始化 ---
    cell_idx.x = static_cast<int>(floorf(start_pos.x / grid.cell_size.x));
    if (ray.direction.x >= 0) {
        step.x = 1;
        t_delta.x = grid.cell_size.x / ray.direction.x;
        t_max.x = ((cell_idx.x + 1) * grid.cell_size.x - start_pos.x) / ray.direction.x;
    }
    else {
        step.x = -1;
        t_delta.x = grid.cell_size.x / -ray.direction.x;
        t_max.x = (cell_idx.x * grid.cell_size.x - start_pos.x) / ray.direction.x;
    }

    // --- Y轴初始化 ---
    cell_idx.y = static_cast<int>(floorf(start_pos.y / grid.cell_size.y));
    if (ray.direction.y >= 0) {
        step.y = 1;
        t_delta.y = grid.cell_size.y / ray.direction.y;
        t_max.y = ((cell_idx.y + 1) * grid.cell_size.y - start_pos.y) / ray.direction.y;
    }
    else {
        step.y = -1;
        t_delta.y = grid.cell_size.y / -ray.direction.y;
        t_max.y = (cell_idx.y * grid.cell_size.y - start_pos.y) / ray.direction.y;
    }

    // --- Z轴初始化 ---
    cell_idx.z = static_cast<int>(floorf(start_pos.z / grid.cell_size.z));
    if (ray.direction.z >= 0) {
        step.z = 1;
        t_delta.z = grid.cell_size.z / ray.direction.z;
        t_max.z = ((cell_idx.z + 1) * grid.cell_size.z - start_pos.z) / ray.direction.z;
    }
    else {
        step.z = -1;
        t_delta.z = grid.cell_size.z / -ray.direction.z;
        t_max.z = (cell_idx.z * grid.cell_size.z - start_pos.z) / ray.direction.z;
    }

    // --- 2. 遍历循环 ---
    while (true) {
        if (cell_idx.x >= 0 && cell_idx.x < grid.grid_dims.x &&
            cell_idx.y >= 0 && cell_idx.y < grid.grid_dims.y &&
            cell_idx.z >= 0 && cell_idx.z < grid.grid_dims.z) {

            int flat_idx = cell_idx.x + cell_idx.y * grid.grid_dims.x + cell_idx.z * grid.grid_dims.x * grid.grid_dims.y;
            int start = grid.d_cell_starts[flat_idx];
            int end = grid.d_cell_starts[flat_idx + 1];

            for (int i = start; i < end; ++i) {
                int h_id = grid.d_cell_entries[i];
                if (h_id == originating_h_id) {
                    continue;
                }

                float t_intersection;
                if (intersectRayPlane(ray, d_heliostats[h_id], t_intersection)) {
                    return true;
                }
            }
        }
        else {
            return false;
        }

        // DDA步进到下一个单元格
        if (t_max.x < t_max.y) {
            if (t_max.x < t_max.z) {
                cell_idx.x += step.x;
                t_max.x += t_delta.x;
            }
            else {
                cell_idx.z += step.z;
                t_max.z += t_delta.z;
            }
        }
        else {
            if (t_max.y < t_max.z) {
                cell_idx.y += step.y;
                t_max.y += t_delta.y;
            }
            else {
                cell_idx.z += step.z;
                t_max.z += t_delta.z;
            }
        }
    }

    return false;
}

#endif // INTERSECTIONS_CUH