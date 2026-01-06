#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <fstream>

#include <cuda_runtime.h>

#include "vector_math.cuh"
#include "simulation.cuh"
#include "kernel.cuh"

#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// ==================================================================================
// 仿真配置
// ==================================================================================
namespace SimConfig {
    const std::string HELIOSTAT_DATA_FILE = "data.csv";
    const float LATITUDE = 39.4f;
    const float LONGITUDE = 98.5f;
    const float ALTITUDE_KM = 3.0f;
    const float HELIOSTAT_WIDTH = 6.0f;
    const float HELIOSTAT_HEIGHT = 6.0f;
    const float RECEIVER_RADIUS = 3.5f;
    const float RECEIVER_HEIGHT = 8.0f;
    const float3 AIM_POINT = make_float3(0.0f, 0.0f, 80.0f);
    const float3 RECEIVER_GEOMETRY_BASE = make_float3(0.0f, 0.0f, 76.0f);
    const float MIRROR_REFLECTIVITY = 0.92f;
    const float SUN_CONE_HALF_ANGLE_MRAD = 4.65f;
    const float NORMAL_PERTURBATION_SIGMA_MRAD = 1.0f;
    const int RAYS_PER_MICROFACET = 512;
    const float MICROFACET_SIZE = 0.01f; // 6.0m / 600 = 0.01m
}

// [用户指定] 在这里设置您想要详细分析的定日镜ID (ID从0开始)
const int HELIOSTAT_ID_TO_INSPECT = 1010;

// ==================================================================================
// 辅助函数
// ==================================================================================
float calculateDNI(float altitude_km, float sun_altitude_rad) {
    if (sun_altitude_rad <= 0) return 0.0f;
    const float G0 = 1.366f;
    float a = 0.4237f - 0.00821f * powf(6.0f - altitude_km, 2);
    float b = 0.5055f + 0.00595f * powf(6.5f - altitude_km, 2);
    float c = 0.2711f + 0.01858f * powf(2.5f - altitude_km, 2);
    float sin_alpha_s = sinf(sun_altitude_rad);
    if (sin_alpha_s < 1e-6) return 0.0f;
    return G0 * (a + b * expf(-c / sin_alpha_s));
}

static inline float degreesToRadians(float degrees) {
    return degrees * 3.1415926535f / 180.0f;
}


// ==================================================================================
// 主函数
// ==================================================================================
int main(int argc, char** argv) {
    std::cout << "======================================================" << std::endl;
    std::cout << " Tower Solar Power Plant Simulation - 2023-A-Q1" << std::endl;
    std::cout << "======================================================" << std::endl;

    std::cout << "\n[Phase 1&2] Initializing CPU/GPU resources..." << std::endl;
    std::vector<Heliostat> h_heliostats;
    loadHeliostatData(SimConfig::HELIOSTAT_DATA_FILE, h_heliostats, SimConfig::HELIOSTAT_WIDTH, SimConfig::HELIOSTAT_HEIGHT);
    const int num_heliostats = h_heliostats.size();
    if (num_heliostats == 0) { std::cerr << "Error: No data loaded." << std::endl; return -1; }
    std::cout << "-> Loaded " << num_heliostats << " heliostats." << std::endl;

    // [检查] 确保要分析的ID有效
    if (HELIOSTAT_ID_TO_INSPECT >= num_heliostats) {
        std::cerr << "Warning: HELIOSTAT_ID_TO_INSPECT (" << HELIOSTAT_ID_TO_INSPECT
            << ") is out of bounds. Max ID is " << num_heliostats - 1
            << ". Detailed analysis will be skipped." << std::endl;
    }

    AccelerationGrid h_grid;
    initializeAccelerationGrid(h_heliostats, h_grid, make_float3(10.f, 10.f, 10.f));
    std::cout << "-> Acceleration grid structure initialized." << std::endl;

    Receiver h_receiver;
    h_receiver.center = SimConfig::RECEIVER_GEOMETRY_BASE;
    h_receiver.radius = SimConfig::RECEIVER_RADIUS;
    h_receiver.height = SimConfig::RECEIVER_HEIGHT;

    Heliostat* d_heliostats;
    Receiver* d_receiver;
    AccelerationGrid* d_grid_struct;
    HeliostatStats* d_stats; // [修改] 使用新的详细统计数组
    CHECK_CUDA_ERROR(cudaMalloc(&d_heliostats, num_heliostats * sizeof(Heliostat)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_receiver, sizeof(Receiver)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_grid_struct, sizeof(AccelerationGrid)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_stats, num_heliostats * sizeof(HeliostatStats))); // [修改] 分配内存
    CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_starts, h_grid.cpu_cell_starts.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_entries, h_grid.cpu_cell_entries.size() * sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_receiver, &h_receiver, sizeof(Receiver), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_starts, h_grid.cpu_cell_starts.data(), h_grid.cpu_cell_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_entries, h_grid.cpu_cell_entries.data(), h_grid.cpu_cell_entries.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_grid_struct, &h_grid, sizeof(AccelerationGrid), cudaMemcpyHostToDevice));

    const int microfacets_per_heliostat_x = static_cast<int>(SimConfig::HELIOSTAT_WIDTH / SimConfig::MICROFACET_SIZE);
    const int num_microfacets_per_helio = microfacets_per_heliostat_x * microfacets_per_heliostat_x;
    const unsigned long long total_microfacets = (unsigned long long)num_heliostats * num_microfacets_per_helio;
    const unsigned long long total_rays_to_cast_per_timepoint = total_microfacets * SimConfig::RAYS_PER_MICROFACET;

    std::cout << "\n[Phase 3/5] Starting simulation loop..." << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    std::ofstream ray_log_file("ray_stats.log");
    unsigned long long grand_total_rays_cast = 0;
    unsigned long long grand_total_shadow_blocked = 0;
    unsigned long long grand_total_hit = 0;

    std::map<int, std::vector<double>> monthly_cosine_eff, monthly_sb_eff, monthly_trunc_eff, monthly_optical_eff, monthly_power_per_area;

    const int months[] = { 5 };
    const float hours[] = { 9.0f };

    int total_timepoints_to_run = sizeof(months) / sizeof(int) * sizeof(hours) / sizeof(float);
    int current_timepoint = 0;

    for (int month : months) {
        for (float hour : hours) {
            current_timepoint++;
            std::cout << "\r-> Simulating... [" << std::setw(3) << current_timepoint << "/" << total_timepoints_to_run << "] "
                << "Month: " << std::setw(2) << month << ", Hour: " << std::fixed << std::setprecision(2) << hour << std::flush;
            float3 sun_direction = calculateSunPosition(month, 21, hour, SimConfig::LATITUDE, SimConfig::LONGITUDE);
            float sun_altitude_rad = asinf(sun_direction.z);
            float current_dni = calculateDNI(SimConfig::ALTITUDE_KM, sun_altitude_rad);

            for (int i = 0; i < num_heliostats; ++i) {
                float3 to_receiver = normalize(SimConfig::AIM_POINT - h_heliostats[i].center);
                h_heliostats[i].ideal_normal = normalize(sun_direction + to_receiver);
            }

            populateAccelerationGrid(h_heliostats, h_grid);
            if (h_grid.d_cell_entries) CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_entries));
            CHECK_CUDA_ERROR(cudaMalloc(&h_grid.d_cell_entries, h_grid.cpu_cell_entries.size() * sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_starts, h_grid.cpu_cell_starts.data(), h_grid.cpu_cell_starts.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(h_grid.d_cell_entries, h_grid.cpu_cell_entries.data(), h_grid.cpu_cell_entries.size() * sizeof(int), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_grid_struct, &h_grid, sizeof(AccelerationGrid), cudaMemcpyHostToDevice));
            CHECK_CUDA_ERROR(cudaMemcpy(d_heliostats, h_heliostats.data(), num_heliostats * sizeof(Heliostat), cudaMemcpyHostToDevice));

            // [修改] 在每次kernel调用前，将统计数组清零
            CHECK_CUDA_ERROR(cudaMemset(d_stats, 0, num_heliostats * sizeof(HeliostatStats)));

            const int threads_per_block = 256;
            const int blocks_per_grid = (total_microfacets + threads_per_block - 1) / threads_per_block;

            // --- 1. GPU 性能计时 ---
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            CHECK_CUDA_ERROR(cudaEventRecord(start));

            rayTracingKernel << <blocks_per_grid, threads_per_block >> > (
                d_heliostats, num_heliostats, d_receiver, d_grid_struct, sun_direction,
                d_stats, // [修改] 传入新的统计数组
                SimConfig::RAYS_PER_MICROFACET, microfacets_per_heliostat_x,
                SimConfig::SUN_CONE_HALF_ANGLE_MRAD / 1000.0f,
                SimConfig::NORMAL_PERTURBATION_SIGMA_MRAD / 1000.0f
                );

            CHECK_CUDA_ERROR(cudaEventRecord(stop));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

            // --- 2. 计算并报告平均追踪速度 ---
            float kernel_execution_time_ms = 0;
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&kernel_execution_time_ms, start, stop));

            const unsigned long long rays_per_heliostat = (unsigned long long)num_microfacets_per_helio * SimConfig::RAYS_PER_MICROFACET;
            const double time_per_heliostat_ms = (double)kernel_execution_time_ms / num_heliostats;
            const double target_ray_count = 100000000.0;
            const double normalized_speed_ms = time_per_heliostat_ms * (target_ray_count / rays_per_heliostat);

            std::cout << "\n\n--- [GPU Ray Tracing Performance Analysis] ---" << std::endl;
            std::cout << "Target Time Point               : Month=" << month << ", Hour=" << hour << std::endl;
            std::cout << "Total Kernel Execution Time     : " << kernel_execution_time_ms << " ms (for " << num_heliostats << " heliostats)" << std::endl;
            std::cout << "Rays Cast per Heliostat         : " << rays_per_heliostat << " (" << (double)rays_per_heliostat / 1e6 << " million)" << std::endl;
            std::cout << "Average Time per Heliostat      : " << std::fixed << std::setprecision(4) << time_per_heliostat_ms << " ms" << std::endl;
            std::cout << "----------------------------------------------------------------" << std::endl;
            std::cout << ">> Normalized Speed per Heliostat: " << std::fixed << std::setprecision(4) << normalized_speed_ms << " ms per 100 million rays" << std::endl;
            std::cout << "----------------------------------------------------------------\n" << std::endl;

            // --- 3. 获取所有统计数据并分析指定定日镜 ---
            std::vector<HeliostatStats> h_stats(num_heliostats);
            CHECK_CUDA_ERROR(cudaMemcpy(h_stats.data(), d_stats, num_heliostats * sizeof(HeliostatStats), cudaMemcpyDeviceToHost));

            if (HELIOSTAT_ID_TO_INSPECT >= 0 && HELIOSTAT_ID_TO_INSPECT < num_heliostats) {
                const HeliostatStats& stats = h_stats[HELIOSTAT_ID_TO_INSPECT];
                const Heliostat& h = h_heliostats[HELIOSTAT_ID_TO_INSPECT];
                std::cout << "--- [Detailed Analysis for Heliostat ID: " << HELIOSTAT_ID_TO_INSPECT << "] ---" << std::endl;
                std::cout << "Position (X, Y, Z)            : (" << h.center.x << ", " << h.center.y << ", " << h.center.z << ")" << std::endl;
                std::cout << "Total Rays Cast               : " << rays_per_heliostat << std::endl;
                std::cout << "----------------------------------------------------------------" << std::endl;
                std::cout << "Shadowed Rays                 : " << std::setw(12) << stats.shadow_rays << " (" << std::fixed << std::setprecision(2) << (double)stats.shadow_rays * 100.0 / rays_per_heliostat << "%)" << std::endl;
                std::cout << "Blocked Rays (post-reflection): " << std::setw(12) << stats.blocked_rays << " (" << std::fixed << std::setprecision(2) << (double)stats.blocked_rays * 100.0 / rays_per_heliostat << "%)" << std::endl;
                std::cout << "Hit Receiver Rays             : " << std::setw(12) << stats.hit_rays << " (" << std::fixed << std::setprecision(2) << (double)stats.hit_rays * 100.0 / rays_per_heliostat << "%)" << std::endl;
                std::cout << "----------------------------------------------------------------\n" << std::endl;
            }

            // --- 4. [修改] 汇总当前时间点的总光线数，用于效率计算和日志 ---
            unsigned long long step_total_shadow = 0, step_total_blocked = 0, step_total_hit = 0;
            for (const auto& s : h_stats) {
                step_total_shadow += s.shadow_rays;
                step_total_blocked += s.blocked_rays;
                step_total_hit += s.hit_rays;
            }

            grand_total_rays_cast += total_rays_to_cast_per_timepoint;
            grand_total_shadow_blocked += step_total_shadow + step_total_blocked;
            grand_total_hit += step_total_hit;

            ray_log_file << "Month: " << month << ", Hour: " << hour << "\n"
                << "  Total Rays: " << total_rays_to_cast_per_timepoint << "\n"
                << "  Shadow Rays: " << step_total_shadow << "\n"
                << "  Blocked Rays: " << step_total_blocked << "\n"
                << "  Hit Rays: " << step_total_hit << "\n\n";

            // --- 5. [修改] 使用汇总后的精确数据计算各项效率 ---
            unsigned long long valid_rays_for_trunc_test = total_rays_to_cast_per_timepoint - (step_total_shadow + step_total_blocked);

            double cosine_eff = 0;
            for (const auto& h : h_heliostats) cosine_eff += dot(sun_direction, h.ideal_normal);
            cosine_eff = (num_heliostats > 0) ? cosine_eff / num_heliostats : 0.0;

            double sb_eff = (total_rays_to_cast_per_timepoint > 0) ? (double)valid_rays_for_trunc_test / total_rays_to_cast_per_timepoint : 0.0;
            double trunc_eff = (valid_rays_for_trunc_test > 0) ? (double)step_total_hit / valid_rays_for_trunc_test : 0.0;

            double total_atm_eff = 0.0;
            if (num_heliostats > 0) for (const auto& h : h_heliostats) {
                float d_HR = length(h.center - SimConfig::AIM_POINT);
                total_atm_eff += (d_HR <= 1000.0f) ? (0.99321 - 0.0001176 * d_HR + 1.97e-8 * d_HR * d_HR) : (0.99321 - 0.0001176 * 1000.0 + 1.97e-8 * 1000.0 * 1000.0);
            }
            double avg_atm_eff = (num_heliostats > 0) ? total_atm_eff / num_heliostats : 0.0;

            double optical_eff = cosine_eff * sb_eff * trunc_eff * avg_atm_eff * SimConfig::MIRROR_REFLECTIVITY;
            double current_power_per_area = current_dni * optical_eff;

            monthly_cosine_eff[month].push_back(cosine_eff);
            monthly_sb_eff[month].push_back(sb_eff);
            monthly_trunc_eff[month].push_back(trunc_eff);
            monthly_optical_eff[month].push_back(optical_eff);
            monthly_power_per_area[month].push_back(current_power_per_area);
        }
    }

    ray_log_file << "========================================\n"
        << "           Grand Total Summary          \n"
        << "========================================\n"
        << "Total Rays Cast Across All Time Points : " << grand_total_rays_cast << "\n"
        << "Total Shadowed/Blocked Rays          : " << grand_total_shadow_blocked << "\n"
        << "Total Hit Receiver Rays              : " << grand_total_hit << "\n";
    ray_log_file.close();
    std::cout << "\nSimulation loop finished." << std::endl;

    std::cout << "\n[Phase 4/5] Writing results to results.csv..." << std::endl;
    // ... [后续结果输出到CSV文件的代码保持不变，因为它依赖于月度数据，而月度数据现在是正确的] ...
    std::ofstream csv_file("results.csv");
    csv_file << std::fixed << std::setprecision(4);
    csv_file << "Table 1: Monthly Averages\n";
    csv_file << "Date,Avg. Optical Eff.,Avg. Cosine Eff.,Avg. S/B Eff.,Avg. Truncation Eff.,Power per Area (kW/m^2)\n";
    double year_total_cosine = 0, year_total_sb = 0, year_total_trunc = 0, year_total_optical = 0, year_total_power = 0;
    for (int month : months) {
        double month_sum_cos = 0, month_sum_sb = 0, month_sum_trunc = 0, month_sum_optical = 0, month_sum_power = 0;
        int count = monthly_cosine_eff[month].size();
        for (double val : monthly_cosine_eff[month]) month_sum_cos += val;
        for (double val : monthly_sb_eff[month]) month_sum_sb += val;
        for (double val : monthly_trunc_eff[month]) month_sum_trunc += val;
        for (double val : monthly_optical_eff[month]) month_sum_optical += val;
        for (double val : monthly_power_per_area[month]) month_sum_power += val;
        double avg_opt = (count > 0) ? month_sum_optical / count : 0.0;
        double avg_cos = (count > 0) ? month_sum_cos / count : 0.0;
        double avg_sb = (count > 0) ? month_sum_sb / count : 0.0;
        double avg_trunc = (count > 0) ? month_sum_trunc / count : 0.0;
        double avg_power = (count > 0) ? month_sum_power / count : 0.0;
        csv_file << month << "/21," << avg_opt << "," << avg_cos << "," << avg_sb << "," << avg_trunc << "," << avg_power << "\n";
        year_total_optical += month_sum_optical; year_total_cosine += month_sum_cos; year_total_sb += month_sum_sb; year_total_trunc += month_sum_trunc; year_total_power += month_sum_power;
    }
    double annual_avg_optical = (total_timepoints_to_run > 0) ? year_total_optical / total_timepoints_to_run : 0.0;
    double annual_avg_cosine = (total_timepoints_to_run > 0) ? year_total_cosine / total_timepoints_to_run : 0.0;
    double annual_avg_sb = (total_timepoints_to_run > 0) ? year_total_sb / total_timepoints_to_run : 0.0;
    double annual_avg_trunc = (total_timepoints_to_run > 0) ? year_total_trunc / total_timepoints_to_run : 0.0;
    double annual_avg_power_per_area = (total_timepoints_to_run > 0) ? year_total_power / total_timepoints_to_run : 0.0;
    double total_mirror_area = num_heliostats * SimConfig::HELIOSTAT_WIDTH * SimConfig::HELIOSTAT_HEIGHT;
    double annual_avg_total_power_MW = annual_avg_power_per_area * total_mirror_area / 1000.0;
    csv_file << "\n\nTable 2: Annual Averages\n";
    csv_file << "Metric,Value\n";
    csv_file << "Year Avg Optical Efficiency," << annual_avg_optical << "\n";
    csv_file << "Year Avg Cosine Efficiency," << annual_avg_cosine << "\n";
    csv_file << "Year Avg Shadow/Block Efficiency," << annual_avg_sb << "\n";
    csv_file << "Year Avg Truncation Efficiency," << annual_avg_trunc << "\n";
    csv_file << "Year Avg Heat Power (MW)," << annual_avg_total_power_MW << "\n";
    csv_file << "Year Avg Power per Area (kW/m^2)," << annual_avg_power_per_area << "\n";
    csv_file.close();
    std::cout << "-> Results successfully written to results.csv." << std::endl;
    std::cout << "-> Detailed ray statistics saved to ray_stats.log." << std::endl;


    std::cout << "\n[Phase 5/5] Cleaning up..." << std::endl;
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_heliostats));
    CHECK_CUDA_ERROR(cudaFree(d_receiver));
    CHECK_CUDA_ERROR(cudaFree(d_grid_struct));
    CHECK_CUDA_ERROR(cudaFree(d_stats)); // [修改] 释放新的统计数组
    CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_starts));
    CHECK_CUDA_ERROR(cudaFree(h_grid.d_cell_entries));
    std::cout << "\nSimulation finished successfully." << std::endl;

    return 0;
}