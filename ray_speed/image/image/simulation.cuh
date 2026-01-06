#ifndef SIMULATION_CUH
#define SIMULATION_CUH

#include <vector>
#include <string>
#include "vector_math.cuh" // 引入基础向量类型和运算

// ==================================================================================
// 核心物理实体数据结构
// ==================================================================================

/**
 * @struct Heliostat
 * @brief 描述单个定日镜的所有属性。
 * 包含了其物理位置、尺寸以及随时间变化的朝向（法向量）。
 * 这个结构将被大量创建并拷贝到GPU上。
 */
struct Heliostat {
    int id;                 // 定日镜的唯一标识符，从0开始
    float3 center;          // 镜面中心在世界坐标系中的位置 (x, y, 安装高度z)
    float width;            // 镜面宽度 (沿其局部x轴)
    float height;           // 镜面高度 (沿其局部y轴)

    // --- 以下为每个时间点更新的动态数据 ---
    float3 ideal_normal;    // 镜面的理想法向量，用于将太阳中心光线反射到吸收塔中心
};

/**
 * @struct Receiver
 * @brief 描述吸收塔（接收器）的几何属性。
 * 在本次仿真中，它是一个位于塔顶的圆柱体。
 */
struct Receiver {
    float3 center;          // 圆柱体底面中心的坐标
    float radius;           // 圆柱体的半径
    float height;           // 圆柱体的高度

    // --- [新增] 用于接收器表面通量图的分辨率 ---
    int grid_width;         // 角度方向的格子数量
    int grid_height;        // 高度方向的格子数量
};

// ==================================================================================
// GPU加速与统计数据结构
// ==================================================================================

/**
 * @struct AccelerationGrid
 * @brief 3D-DDA加速网格的数据结构。
 *
 * 该结构在CPU端构建，然后将其元数据和指向GPU内存的指针上传到GPU。
 * 它将整个定日镜场空间划分为均匀的单元格，以加速光线求交测试。
 */
struct AccelerationGrid {
    // --- 网格元数据 (在CPU上计算，然后拷贝到GPU) ---
    float3 world_min;       // 整个场地有效包围盒的最小角点坐标
    float3 cell_size;       // 每个小格子的尺寸 (x, y, z)
    int3   grid_dims;       // 网格在X, Y, Z三个方向上的格子数量

    // --- 指向GPU内存的指针 ---
    // 这些指针在CPU端被赋值为cudaMalloc返回的地址，
    // 然后整个结构体被拷贝到GPU，使得GPU代码可以通过这些指针访问数据。
    int* d_cell_starts;     // 每个格子的起始索引 (指向 d_cell_entries)。数组长度为 grid_dims.x*y*z + 1。
    int* d_cell_entries;    // 存储所有格子里包含的定日镜ID的“巨型”数组。

    // --- 仅用于CPU端构建的临时数据 ---
    // 这些数据在构建完成后，用于向GPU拷贝数据，之后可以被释放。
    std::vector<int> cpu_cell_starts;
    std::vector<int> cpu_cell_entries;
};

/**
 * @struct AtomicCounters
 * @brief 用于在GPU上进行线程安全计数的结构体。 [保留，但不再是主要统计工具]
 *
 * 每个成员都将使用atomicAdd进行更新，以避免并行写入冲突。
 * 使用64位无符号整数以防止因海量光线追踪而导致的溢出。
 */
struct AtomicCounters {
    unsigned long long shadow_rays;      // 因被其他定日镜遮挡太阳而损失的光线总数
    unsigned long long blocked_rays;     // 反射后因被其他定日镜阻挡而损失的光线总数
    unsigned long long hit_rays;         // 成功击中吸收器的光线总数
};

/**
 * @struct HeliostatStats
 * @brief [新增] 用于存储单个定日镜的光线追踪统计数据。
 *
 * 这是我们实现精细化分析的关键。我们将为每个定日镜在GPU上创建一个此结构的实例。
 */
struct HeliostatStats {
    unsigned long long shadow_rays;
    unsigned long long blocked_rays;
    unsigned long long hit_rays;
};


// [新增] 用于存储每个微面元遮挡数据的结构体
struct MicrofacetShadingData {
    // 使用 unsigned int 来节省空间，因为一个微面元的光线数不会超过512
    unsigned int shadow_ray_losses;
    unsigned int block_ray_losses;
};

// ==================================================================================
// CPU端辅助函数的声明 (函数实现在 simulation.cu)
// ==================================================================================

/**
 * @brief 从CSV文件加载定日镜的位置数据。
 * @param filename 输入的CSV文件名 (例如 "data.csv")。
 * @param heliostats_out [输出] 用于存储加载的定日镜对象的向量。
 * @param width 所有定日镜的宽度。
 * @param height 所有定日镜的高度。
 */
void loadHeliostatData(const std::string& filename, std::vector<Heliostat>& heliostats_out, float width, float height);

/**
 * @brief 构建3D-DDA加速网格。
 * @param heliostats [输入] 包含所有定日镜信息的向量。
 * @param grid_out [输出] 用于存储构建好的加速网格对象的引用。
 * @param cell_size 每个网格单元的尺寸。
 */
 // [重构] 删除旧的函数声明
 // void buildAccelerationGrid(const std::vector<Heliostat>& heliostats, AccelerationGrid& grid_out, float3 cell_size);

 // [重构] 添加新的函数声明
void initializeAccelerationGrid(const std::vector<Heliostat>& heliostats, AccelerationGrid& grid_out, float3 cell_size);
void populateAccelerationGrid(const std::vector<Heliostat>& heliostats, AccelerationGrid& grid);
float3 calculateSunPosition(int month, int day, float local_solar_time, float latitude, float longitude);


#endif // SIMULATION_CUH