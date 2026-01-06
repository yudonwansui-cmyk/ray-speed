/**
 * @file simulation.cu
 * @brief [重构版] 实现仿真场景的核心CPU端辅助函数。
 *
 * 包含了加载数据、计算太阳位置，以及重构后的、支持动态更新的
 * 3D-DDA加速网格构建逻辑。
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "simulation.cuh"

 // 定义PI常量
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 辅助函数：将角度从度转换为弧度
static inline float degreesToRadians(float degrees) {
    return degrees * M_PI / 180.0f;
}

// ==================================================================================
// 函数实现: calculateSunPosition (保持最终修正版)
// ==================================================================================
float3 calculateSunPosition(int month, int day, float local_solar_time, float latitude, float longitude) {
    const float phi = degreesToRadians(latitude);
    const int days_before_month[] = { 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334 };
    int day_of_year = days_before_month[month - 1] + day;
    int D = day_of_year - 80;

    const float declination_angle_rad = degreesToRadians(23.45f);
    const float delta = asinf(sinf(declination_angle_rad) * sinf(2.0f * M_PI * D / 365.0f));
    const float omega = degreesToRadians(15.0f * (local_solar_time - 12.0f));
    const float sin_alpha_s = cosf(delta) * cosf(phi) * cosf(omega) + sinf(delta) * sinf(phi);
    const float alpha_s = asinf(sin_alpha_s);

    // --- [修正] 使用标准的、基于(East, North, Up)右手坐标系的太阳矢量公式 ---
    // Z分量 (sin_alpha_s) 已经在前面正确计算，可以直接使用
    float sun_z = sin_alpha_s;

    // X分量 (东向)
    float sun_x = cosf(delta) * sinf(omega);

    // Y分量 (北向)
    float sun_y = cosf(delta) * sinf(phi) * cosf(omega) - sinf(delta) * cosf(phi);
    // --- [修正结束] ---

    return normalize(make_float3(sun_x, sun_y, sun_z));
}


// ==================================================================================
// 函数实现: loadHeliostatData (保持健壮版)
// ==================================================================================
void loadHeliostatData(const std::string& filename, std::vector<Heliostat>& heliostats_out, float width, float height) {
    heliostats_out.clear();
    std::ifstream file(filename);
    if (!file.is_open()) { throw std::runtime_error("Failed to open file: " + filename); }

    std::string line;
    int id_counter = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string value;
        float x, y, z;
        try {
            std::getline(ss, value, ','); x = std::stof(value);
            std::getline(ss, value, ','); y = std::stof(value);
            std::getline(ss, value);      z = std::stof(value);
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Skipping malformed line: \"" << line << "\". Error: " << e.what() << std::endl;
            continue;
        }
        Heliostat h;
        h.id = id_counter++;
        h.center = make_float3(x, y, z);
        h.width = width;
        h.height = height;
        heliostats_out.push_back(h);
    }
    file.close();
}


// ==================================================================================
// 函数实现: 重构后的加速网格构建
// ==================================================================================

/**
 * @brief [重构] 初始化加速网格的元数据和CPU端存储。
 * 这是一个一次性的设置，在主循环开始前调用。
 */
void initializeAccelerationGrid(const std::vector<Heliostat>& heliostats, AccelerationGrid& grid_out, float3 cell_size) {
    if (heliostats.empty()) {
        std::cerr << "Warning: Trying to initialize grid for an empty heliostat field." << std::endl;
        return;
    }

    // --- 1. 计算一个足够大的、能容纳所有可能姿态的全局包围盒 ---
    float3 world_min = make_float3(INFINITY, INFINITY, INFINITY);
    float3 world_max = make_float3(-INFINITY, -INFINITY, -INFINITY);

    for (const auto& h : heliostats) {
        // 使用一个保守的最大尺寸（对角线长度）来确定全局边界
        float max_extent_radius = length(make_float3(h.width, h.height, 0.0f)) * 0.5f;
        world_min.x = std::min(world_min.x, h.center.x - max_extent_radius);
        world_min.y = std::min(world_min.y, h.center.y - max_extent_radius);
        world_min.z = std::min(world_min.z, h.center.z - max_extent_radius);
        world_max.x = std::max(world_max.x, h.center.x + max_extent_radius);
        world_max.y = std::max(world_max.y, h.center.y + max_extent_radius);
        world_max.z = std::max(world_max.z, h.center.z + max_extent_radius);
    }
    // 增加一点安全边界
    world_min = world_min - make_float3(1.0f, 1.0f, 1.0f);
    world_max = world_max + make_float3(1.0f, 1.0f, 1.0f);

    // --- 2. 设置网格元数据 ---
    grid_out.world_min = world_min;
    grid_out.cell_size = cell_size;
    float3 world_size = world_max - world_min;
    grid_out.grid_dims = make_int3(
        static_cast<int>(ceilf(world_size.x / cell_size.x)),
        static_cast<int>(ceilf(world_size.y / cell_size.y)),
        static_cast<int>(ceilf(world_size.z / cell_size.z))
    );

    // --- 3. 初始化CPU端存储 ---
    int num_cells = grid_out.grid_dims.x * grid_out.grid_dims.y * grid_out.grid_dims.z;
    grid_out.cpu_cell_starts.resize(num_cells + 1);
    // 为频繁的clear-insert操作预留空间
    grid_out.cpu_cell_entries.reserve(heliostats.size() * 20); // 假设平均每个镜子占据20个格子
}


/**
 * @brief [新增] 根据当前时刻的定日镜姿态，动态地、精确地填充加速网格。
 * 这个函数将在每个时间点被调用。
 */
void populateAccelerationGrid(const std::vector<Heliostat>& heliostats, AccelerationGrid& grid) {
    int num_cells = grid.grid_dims.x * grid.grid_dims.y * grid.grid_dims.z;
    std::vector<std::vector<int>> temp_cells(num_cells);

    // --- 遍历每个定日镜，计算其紧凑AABB并填充格子 ---
    for (const auto& h : heliostats) {
        // 1. 构建当前姿态的局部坐标系
      // [最终修正] 使用与GPU端完全一致的、稳健的方法来构建坐标系
        float3 local_z_axis = h.ideal_normal;
        float3 up_vector = make_float3(0.0f, 0.0f, 1.0f);
        float3 local_x_axis, local_y_axis;

        if (abs(dot(local_z_axis, up_vector)) > 0.9999f) {
            local_x_axis = make_float3(1.0f, 0.0f, 0.0f);
            local_y_axis = normalize(cross(local_z_axis, local_x_axis));
        }
        else {
            local_x_axis = normalize(cross(local_z_axis, up_vector));
            local_y_axis = normalize(cross(local_x_axis, local_z_axis));
        }

        // 2. 计算AABB的“半径” (沿世界坐标轴的投影范围)
        float half_w = h.width * 0.5f;
        float half_h = h.height * 0.5f;
        float r_x = half_w * std::abs(local_x_axis.x) + half_h * std::abs(local_y_axis.x);
        float r_y = half_w * std::abs(local_x_axis.y) + half_h * std::abs(local_y_axis.y);
        float r_z = half_w * std::abs(local_x_axis.z) + half_h * std::abs(local_y_axis.z);

        // 3. 确定紧凑AABB的min和max点
        float3 h_min = h.center - make_float3(r_x, r_y, r_z);
        float3 h_max = h.center + make_float3(r_x, r_y, r_z);

        // 4. 将其包围盒转换为网格索引范围并填充
        int3 start_idx = make_int3(
            std::max(0, static_cast<int>(floorf((h_min.x - grid.world_min.x) / grid.cell_size.x))),
            std::max(0, static_cast<int>(floorf((h_min.y - grid.world_min.y) / grid.cell_size.y))),
            std::max(0, static_cast<int>(floorf((h_min.z - grid.world_min.z) / grid.cell_size.z)))
        );
        int3 end_idx = make_int3(
            std::min(grid.grid_dims.x - 1, static_cast<int>(floorf((h_max.x - grid.world_min.x) / grid.cell_size.x))),
            std::min(grid.grid_dims.y - 1, static_cast<int>(floorf((h_max.y - grid.world_min.y) / grid.cell_size.y))),
            std::min(grid.grid_dims.z - 1, static_cast<int>(floorf((h_max.z - grid.world_min.z) / grid.cell_size.z)))
        );

        for (int z = start_idx.z; z <= end_idx.z; ++z) {
            for (int y = start_idx.y; y <= end_idx.y; ++y) {
                for (int x = start_idx.x; x <= end_idx.x; ++x) {
                    temp_cells[x + y * grid.grid_dims.x + z * grid.grid_dims.x * grid.grid_dims.y].push_back(h.id);
                }
            }
        }
    }

    // --- 扁平化数据以便上传GPU ---
    grid.cpu_cell_entries.clear();
    int current_offset = 0;
    for (int i = 0; i < num_cells; ++i) {
        grid.cpu_cell_starts[i] = current_offset;
        grid.cpu_cell_entries.insert(grid.cpu_cell_entries.end(), temp_cells[i].begin(), temp_cells[i].end());
        current_offset += temp_cells[i].size();
    }
    grid.cpu_cell_starts[num_cells] = current_offset;
}