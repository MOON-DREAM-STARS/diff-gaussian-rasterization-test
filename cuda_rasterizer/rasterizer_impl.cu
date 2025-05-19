/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

// 辅助函数：将设备数据保存到CSV文件
void saveDeviceDataToCSV(const std::string &filename, const float *d_data, int num_elements, int dims_per_point)
{
	// 分配主机内存
	float *h_data = new float[num_elements];

	// 从设备复制到主机
	cudaMemcpy(h_data, d_data, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

	// 创建并打开CSV文件
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		delete[] h_data;
		return;
	}

	// 写入数据
	int points = num_elements / dims_per_point;
	for (int i = 0; i < points; i++)
	{
		for (int j = 0; j < dims_per_point; j++)
		{
			file << std::fixed << std::setprecision(6) << h_data[i * dims_per_point + j];
			if (j < dims_per_point - 1)
			{
				file << ",";
			}
		}
		file << "\n";
	}

	// 关闭文件并释放内存
	file.close();
	delete[] h_data;

	std::cout << "Saved " << points << " points with " << dims_per_point
			  << " dimensions to " << filename << std::endl;
}

// 辅助函数：保存duplicateWithKeys前后的数据到单个CSV文件
void saveDuplicateWithKeysData(
	const std::string &filename,
	const float2 *points_xy,
	const float *depths,
	const uint32_t *offsets,
	const int *radii,
	const uint64_t *keys_unsorted,
	const uint32_t *values_unsorted,
	int P,
	int num_rendered)
{
	// 分配主机内存
	float2 *h_points_xy = new float2[P];
	float *h_depths = new float[P];
	uint32_t *h_offsets = new uint32_t[P];
	int *h_radii = new int[P];

	// 从设备复制基础数据到主机
	cudaMemcpy(h_points_xy, points_xy, P * sizeof(float2), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_depths, depths, P * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_offsets, offsets, P * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_radii, radii, P * sizeof(int), cudaMemcpyDeviceToHost);

	// 准备键值对数据
	uint64_t *h_keys = nullptr;
	uint32_t *h_values = nullptr;

	if (keys_unsorted != nullptr && values_unsorted != nullptr && num_rendered > 0)
	{
		h_keys = new uint64_t[num_rendered];
		h_values = new uint32_t[num_rendered];

		// 确保CUDA操作完成
		cudaDeviceSynchronize();

		// 从设备复制键值对数据到主机
		cudaError_t keysCopyResult = cudaMemcpy(h_keys, keys_unsorted, num_rendered * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaError_t valuesCopyResult = cudaMemcpy(h_values, values_unsorted, num_rendered * sizeof(uint32_t), cudaMemcpyDeviceToHost);

		if (keysCopyResult != cudaSuccess || valuesCopyResult != cudaSuccess)
		{
			std::cerr << "CUDA error copying keys/values: "
					  << cudaGetErrorString(keysCopyResult) << " / "
					  << cudaGetErrorString(valuesCopyResult) << std::endl;

			// 出错时释放内存
			delete[] h_keys;
			delete[] h_values;
			h_keys = nullptr;
			h_values = nullptr;
		}
	}

	// 创建并打开CSV文件
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		goto cleanup;
	}

	// 写入CSV文件头
	file << "point_id,x,y,depth,radius,offset";
	if (h_keys != nullptr && h_values != nullptr)
	{
		file << ",key,tile_id,depth_bits,value";
	}
	file << "\n";

	// 写入基础数据（处理前）
	for (int i = 0; i < P; i++)
	{
		file << i << ","
			 << std::fixed << std::setprecision(6) << h_points_xy[i].x << ","
			 << std::fixed << std::setprecision(6) << h_points_xy[i].y << ","
			 << std::fixed << std::setprecision(6) << h_depths[i] << ","
			 << h_radii[i] << ","
			 << h_offsets[i];
		file << "\n";
	}

	// 如果提供了处理后的数据，则添加这些数据
	if (h_keys != nullptr && h_values != nullptr)
	{
		file << "\n# Processed data (duplicated with keys)\n";
		file << "index,key,tile_id,depth_bits,value\n";

		for (int i = 0; i < num_rendered; i++)
		{
			uint64_t key = h_keys[i];
			uint32_t tile_id = key >> 32;
			uint32_t depth_bits = key & 0xFFFFFFFF;

			file << i << ","
				 << key << ","
				 << tile_id << ","
				 << depth_bits << ","
				 << h_values[i] << "\n";
		}
	}

	file.close();
	std::cout << "Saved duplicateWithKeys data to " << filename << std::endl;

cleanup:
	delete[] h_points_xy;
	delete[] h_depths;
	delete[] h_offsets;
	delete[] h_radii;
	if (h_keys)
		delete[] h_keys;
	if (h_values)
		delete[] h_values;
}

// 辅助函数：保存SortPairs前后的数据到单个CSV文件
void saveSortPairsData(
	const std::string &filename,
	const uint64_t *keys_before,
	const uint32_t *values_before,
	const uint64_t *keys_after,
	const uint32_t *values_after,
	int num_elements)
{
	// 分配主机内存
	uint64_t *h_keys_before = new uint64_t[num_elements];
	uint32_t *h_values_before = new uint32_t[num_elements];

	// 从设备复制数据到主机
	cudaMemcpy(h_keys_before, keys_before, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_values_before, values_before, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);

	// 如果提供了排序后的数据，也复制这些数据
	uint64_t *h_keys_after = nullptr;
	uint32_t *h_values_after = nullptr;

	if (keys_after != nullptr && values_after != nullptr)
	{
		h_keys_after = new uint64_t[num_elements];
		h_values_after = new uint32_t[num_elements];

		cudaMemcpy(h_keys_after, keys_after, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_values_after, values_after, num_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	}

	// 创建并打开CSV文件
	std::ofstream file(filename);
	if (!file.is_open())
	{
		std::cerr << "Error: Could not open file " << filename << std::endl;
		goto cleanup;
	}

	// 写入CSV文件头
	file << "index,before_key,before_tile_id,before_depth_bits,before_value";
	if (h_keys_after != nullptr && h_values_after != nullptr)
	{
		file << ",after_key,after_tile_id,after_depth_bits,after_value";
	}
	file << "\n";

	// 写入数据
	for (int i = 0; i < num_elements; i++)
	{
		uint64_t before_key = h_keys_before[i];
		uint32_t before_tile_id = before_key >> 32;
		uint32_t before_depth_bits = before_key & 0xFFFFFFFF;

		file << i << ","
			 << before_key << ","
			 << before_tile_id << ","
			 << before_depth_bits << ","
			 << h_values_before[i];

		if (h_keys_after != nullptr && h_values_after != nullptr)
		{
			uint64_t after_key = h_keys_after[i];
			uint32_t after_tile_id = after_key >> 32;
			uint32_t after_depth_bits = after_key & 0xFFFFFFFF;

			file << ","
				 << after_key << ","
				 << after_tile_id << ","
				 << after_depth_bits << ","
				 << h_values_after[i];
		}

		file << "\n";
	}

	file.close();
	std::cout << "Saved SortPairs data to " << filename << std::endl;

cleanup:
	delete[] h_keys_before;
	delete[] h_values_before;
	if (h_keys_after)
		delete[] h_keys_after;
	if (h_values_after)
		delete[] h_values_after;
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
							 const float *orig_points,
							 const float *viewmatrix,
							 const float *projmatrix,
							 bool *present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps.
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2 *points_xy,
	const float *depths,
	const uint32_t *offsets,
	uint64_t *gaussian_keys_unsorted,
	uint32_t *gaussian_values_unsorted,
	int *radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t *)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t *point_list_keys, uint2 *ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float *means3D,
	float *viewmatrix,
	float *projmatrix,
	bool *present)
{
	checkFrustum<<<(P + 255) / 256, 256>>>(
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char *&chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char *&chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char *&chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char *(size_t)> geometryBuffer,
	std::function<char *(size_t)> binningBuffer,
	std::function<char *(size_t)> imageBuffer,
	const int P, int D, int M,
	const float *background,
	const int width, int height,
	const float *means3D,
	const float *shs,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float scale_modifier,
	const float *rotations,
	const float *cov3D_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float *out_color,
	float *depth,
	bool antialiasing,
	int *radii,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char *chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char *img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// 保存预处理前的数据
	std::cout << "Saving pre-processing data..." << std::endl;
	saveDeviceDataToCSV("preprocess_before.csv", means3D, P * 3, 3); // 3D点坐标

	// 可以根据需要保存更多输入数据
	if (scales != nullptr)
	{
		saveDeviceDataToCSV("preprocess_before_scales.csv", scales, P * 3, 3);
	}
	if (opacities != nullptr)
	{
		saveDeviceDataToCSV("preprocess_before_opacities.csv", opacities, P, 1);
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
				   P, D, M,
				   means3D,
				   (glm::vec3 *)scales,
				   scale_modifier,
				   (glm::vec4 *)rotations,
				   opacities,
				   shs,
				   geomState.clamped,
				   cov3D_precomp,
				   colors_precomp,
				   viewmatrix, projmatrix,
				   (glm::vec3 *)cam_pos,
				   width, height,
				   focal_x, focal_y,
				   tan_fovx, tan_fovy,
				   radii,
				   geomState.means2D,
				   geomState.depths,
				   geomState.cov3D,
				   geomState.rgb,
				   geomState.conic_opacity,
				   tile_grid,
				   geomState.tiles_touched,
				   prefiltered,
				   antialiasing),
			   debug)

	// 保存预处理后的数据
	std::cout << "Saving post-processing data..." << std::endl;
	saveDeviceDataToCSV("preprocess_after.csv", (float *)geomState.means2D, P * 2, 2);			   // 2D投影点
	saveDeviceDataToCSV("preprocess_after_depths.csv", geomState.depths, P, 1);					   // 深度值
	saveDeviceDataToCSV("preprocess_after_cov3D.csv", geomState.cov3D, P * 6, 6);				   // 3D协方差
	saveDeviceDataToCSV("preprocess_after_rgb.csv", geomState.rgb, P * 3, 3);					   // RGB颜色
	saveDeviceDataToCSV("preprocess_after_conic.csv", (float *)geomState.conic_opacity, P * 4, 4); // 圆锥和不透明度

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char *binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// 保存前的数据 - 在duplicateWithKeys调用之前添加
	std::cout << "Saving duplicateWithKeys input data..." << std::endl;
	saveDuplicateWithKeysData(
		"duplicateWithKeys_before_cuda.csv",
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		radii,
		nullptr, // 还没有处理后的键
		nullptr, // 还没有处理后的值
		P,
		0);

	// For each instance to be rendered, produce adequate [ tile | depth ] key
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys<<<(P + 255) / 256, 256>>>(
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
		CHECK_CUDA(, debug)

		// 确保CUDA操作完成
		cudaDeviceSynchronize();

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// 打印调试信息
	std::cout << "Number of rendered Gaussians: " << num_rendered << std::endl;

	// 保存后的数据 - 在duplicateWithKeys调用之后添加
	std::cout << "Saving duplicateWithKeys output data..." << std::endl;
	saveDuplicateWithKeysData(
		"duplicateWithKeys_after_cuda.csv",
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		radii,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		P,
		num_rendered);

	// 在排序前保存数据
	std::cout << "Saving SortPairs before data..." << std::endl;
	saveSortPairsData(
		"SortPairs_before_cuda.csv",
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		nullptr, // 排序前没有排序后的数据
		nullptr, // 排序前没有排序后的数据
		num_rendered);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
				   binningState.list_sorting_space,
				   binningState.sorting_size,
				   binningState.point_list_keys_unsorted, binningState.point_list_keys,
				   binningState.point_list_unsorted, binningState.point_list,
				   num_rendered, 0, 32 + bit),
			   debug)

	// 在排序后保存数据，包括排序前和排序后的数据
	std::cout << "Saving SortPairs after data..." << std::endl;
	saveSortPairsData(
		"SortPairs_after_cuda.csv",
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		binningState.point_list_keys,
		binningState.point_list,
		num_rendered);

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges<<<(num_rendered + 255) / 256, 256>>>(
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float *feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
				   tile_grid, block,
				   imgState.ranges,
				   binningState.point_list,
				   width, height,
				   geomState.means2D,
				   feature_ptr,
				   geomState.conic_opacity,
				   imgState.accum_alpha,
				   imgState.n_contrib,
				   background,
				   out_color,
				   geomState.depths,
				   depth),
			   debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float *background,
	const int width, int height,
	const float *means3D,
	const float *shs,
	const float *colors_precomp,
	const float *opacities,
	const float *scales,
	const float scale_modifier,
	const float *rotations,
	const float *cov3D_precomp,
	const float *viewmatrix,
	const float *projmatrix,
	const float *campos,
	const float tan_fovx, float tan_fovy,
	const int *radii,
	char *geom_buffer,
	char *binning_buffer,
	char *img_buffer,
	const float *dL_dpix,
	const float *dL_invdepths,
	float *dL_dmean2D,
	float *dL_dconic,
	float *dL_dopacity,
	float *dL_dcolor,
	float *dL_dinvdepth,
	float *dL_dmean3D,
	float *dL_dcov3D,
	float *dL_dsh,
	float *dL_dscale,
	float *dL_drot,
	bool antialiasing,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float *color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	CHECK_CUDA(BACKWARD::render(
				   tile_grid,
				   block,
				   imgState.ranges,
				   binningState.point_list,
				   width, height,
				   background,
				   geomState.means2D,
				   geomState.conic_opacity,
				   color_ptr,
				   geomState.depths,
				   imgState.accum_alpha,
				   imgState.n_contrib,
				   dL_dpix,
				   dL_invdepths,
				   (float3 *)dL_dmean2D,
				   (float4 *)dL_dconic,
				   dL_dopacity,
				   dL_dcolor,
				   dL_dinvdepth),
			   debug);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float *cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
									(float3 *)means3D,
									radii,
									shs,
									geomState.clamped,
									opacities,
									(glm::vec3 *)scales,
									(glm::vec4 *)rotations,
									scale_modifier,
									cov3D_ptr,
									viewmatrix,
									projmatrix,
									focal_x, focal_y,
									tan_fovx, tan_fovy,
									(glm::vec3 *)campos,
									(float3 *)dL_dmean2D,
									dL_dconic,
									dL_dinvdepth,
									dL_dopacity,
									(glm::vec3 *)dL_dmean3D,
									dL_dcolor,
									dL_dcov3D,
									dL_dsh,
									(glm::vec3 *)dL_dscale,
									(glm::vec4 *)dL_drot,
									antialiasing),
			   debug);
}
