#pragma once
#include <cstdint>
#include <sys/mman.h>
#include <omp.h>
#include <vector>
#include <fstream>
#include <cassert>
#include <memory>
#include <cstring>
#include <random>
#include <algorithm>
#include "point.h"

namespace ant
{
    // 从二进制文件加载向量集；文件头格式：[num_vectors][dims][raw data...]
    template <typename IndexType, typename PointRangeType>
    PointRangeType load_point_range(const char *filename, DistanceMetric distance_metric = DistanceMetric::EUCLIDEAN)
    {
        std::ifstream reader(filename, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        IndexType num_vectors, dims;
        using Parameters = typename PointRangeType::Parameters;
        reader.read(reinterpret_cast<char *>(&num_vectors), sizeof(IndexType));
        reader.read(reinterpret_cast<char *>(&dims), sizeof(IndexType));
        auto points = PointRangeType(num_vectors, Parameters(dims, distance_metric));
        points.load(reader);
        reader.close();
        return std::move(points);
    }

    // 内存中的向量集合，每个向量按 64 字节对齐存储
    template <class Point_>
    struct PointRange
    {
        using Point = Point_;
        using Parameters = typename Point::Parameters;
        using byte = uint8_t;
        using T = typename Point::T;

        unsigned dimension() const { return params.dims; }
        size_t capacity() const { return capacity_size; }

        PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)), aligned_bytes(0), num_vectors(0), capacity_size(0) {}

        PointRange(const PointRange &) = default;

        PointRange(size_t num_vectors, const Parameters &params)
            : num_vectors(num_vectors), capacity_size(num_vectors), params(params)
        {
            size_t num_bytes = params.num_bytes();
            aligned_bytes = 64 * ((num_bytes - 1) / 64 + 1);
            size_t total_bytes = capacity_size * aligned_bytes;
            byte *ptr = static_cast<byte *>(aligned_alloc(1l << 21, total_bytes));
            madvise(ptr, total_bytes, MADV_HUGEPAGE);
            values = std::shared_ptr<byte[]>(ptr, std::free);
        }

        // 与 DiskPointRange 保持一致的缓存接口；内存版本为空操作
        void init_cache() {}
        void clear_cache() {}
        void preload(size_t) {}

        template <typename OtherPoint>
        PointRange<Point_> &operator+=(const PointRange<OtherPoint> &other)
        {
#pragma omp parallel for
            for (auto i = 0; i < other.size(); ++i)
            {
                (*this)[i] += other[i];
            }
            return *this;
        }

        PointRange<Point_> &operator*=(float weight)
        {
#pragma omp parallel for
            for (auto i = 0; i < this->size(); ++i)
            {
                (*this)[i] *= weight;
            }
            return *this;
        }

        // 浅拷贝：共享底层内存
        PointRange<Point_> &operator=(const PointRange<Point_> &other)
        {
            ref_copy_(other);
            return *this;
        }

        PointRange<Point_> &operator=(PointRange<Point_> &&other) noexcept
        {
            params = other.params;
            aligned_bytes = other.aligned_bytes;
            capacity_size = other.capacity_size;
            num_vectors = other.num_vectors;
            values = std::move(other.values);
            other.num_vectors = 0;
            other.capacity_size = 0;
            return *this;
        }

        template <typename OtherPoint>
        void add_feat(PointRange<OtherPoint> &other, float weight)
        {
            assert(other.size() == this->size());
#pragma omp parallel for
            for (auto i = 0; i < other.size(); ++i)
            {
                (*this)[i].add_feat(other[i], weight);
            }
        }

        template <typename OtherPoint>
        void add_feat_square(PointRange<OtherPoint> &other, float weight)
        {
            assert(other.size() == this->size());
#pragma omp parallel for
            for (auto i = 0; i < other.size(); ++i)
            {
                (*this)[i].add_feat_square(other[i], weight);
            }
        }

        void copy_(const PointRange<Point_> &other, size_t from_id = 0, size_t to_id = 0, size_t copy_size = static_cast<size_t>(-1))
        {
            if (copy_size == static_cast<size_t>(-1))
            {
                copy_size = num_vectors - to_id;
            }
            while (copy_size > 0)
            {
                memcpy((*this)[to_id++].data(), other[from_id++].data(), params.num_bytes());
                --copy_size;
            }
        }

        void ref_copy_(const PointRange<Point_> &other)
        {
            params = other.params;
            aligned_bytes = other.aligned_bytes;
            capacity_size = other.capacity_size;
            num_vectors = other.num_vectors;
            values = other.values;
        }

        void zeros_()
        {
            memset(values.get(), 0, aligned_bytes * num_vectors);
        }

        // Kaiming 均匀分布初始化（仅 float 向量有效）
        void kaiming_uniform_init(unsigned fan, float negative_slope = 0)
        {
            float bound = std::sqrt(6.0f / ((1.0f + negative_slope * negative_slope) * fan));

            std::random_device rd;

#pragma omp parallel
            {
                std::mt19937 gen(rd() + omp_get_thread_num());
                std::uniform_real_distribution<float> dist(-bound, bound);

#pragma omp for
                for (size_t i = 0; i < num_vectors; ++i)
                {
                    (*this)[i].rand_init(gen, dist);
                }
            }
        }

        // 从已打开的二进制流顺序加载向量
        void load(std::ifstream &reader, size_t start_id = 0)
        {
            constexpr size_t BLOCK_SIZE = 1000000;
            size_t index = 0;
            size_t num_bytes = params.num_bytes();
            std::vector<byte> block_data(BLOCK_SIZE * num_bytes);

            while (start_id + index < num_vectors)
            {
                size_t floor = index;
                size_t ceiling = (index + BLOCK_SIZE <= num_vectors) ? (index + BLOCK_SIZE) : num_vectors;
                long batch_count = static_cast<long>(ceiling - floor);
                byte *data_start = block_data.data();
                reader.read(reinterpret_cast<char *>(data_start), batch_count * num_bytes);

#pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i)
                {
                    std::memmove(values.get() + (i + start_id) * aligned_bytes,
                                 data_start + (i - floor) * num_bytes,
                                 num_bytes);
                }
                index = ceiling;
            }
        }

        // 按 id 列表从文件指定偏移处加载向量
        template <typename IndexType>
        void load(const char *filename, size_t head_offset, const IndexType *ids, size_t num_ids, size_t start_id = 0)
        {
            std::ifstream reader(filename, std::ios::in | std::ios::binary);
            load<IndexType>(reader, head_offset, ids, num_ids, start_id);
            reader.close();
        }

        template <typename IndexType>
        void load(std::ifstream &reader, size_t head_offset, const IndexType *ids, size_t num_ids, size_t start_id = 0, size_t max_batch_size = 100000)
        {
            assert(reader.is_open());
            size_t value_offset = start_id * aligned_bytes;
            size_t num_bytes = params.num_bytes();

            auto max_it = std::max_element(ids, ids + num_ids);
            size_t max_id = (*max_it) + 1;
            std::vector<IndexType> id_to_offset(max_id, static_cast<IndexType>(-1));

            assert(num_ids + start_id <= num_vectors);
            for (size_t i = 0; i < num_ids; ++i)
            {
                IndexType cur_id = ids[i];
                id_to_offset[cur_id] = static_cast<IndexType>(i);
            }

            std::vector<char> cache(max_batch_size * num_bytes);
            reader.seekg(head_offset);

            for (size_t i = 0; i < max_id; i += max_batch_size)
            {
                size_t cur_batch_size = (i + max_batch_size) > max_id ? (max_id - i) : max_batch_size;
                reader.read(cache.data(), cur_batch_size * num_bytes);

                for (size_t j = 0; j < cur_batch_size; ++j)
                {
                    size_t file_id = i + j;
                    if (id_to_offset[file_id] < 0 || id_to_offset[file_id] >= static_cast<IndexType>(max_id))
                    {
                        continue;
                    }

                    auto *src = cache.data() + j * num_bytes;
                    auto *dst = reinterpret_cast<char *>(values.get() + value_offset + id_to_offset[file_id] * aligned_bytes);
                    memcpy(dst, src, num_bytes);
                }
            }
        }

        // 从内存缓存中按 id 映射加载向量
        template <typename IndexType>
        void load(char *cache, const IndexType *ids, size_t num_ids, size_t start_id = 0, size_t max_batch_size = 100000)
        {
            (void)max_batch_size;
            size_t value_offset = start_id * aligned_bytes;
            size_t num_bytes = params.num_bytes();

#pragma omp parallel for
            for (size_t i = 0; i < num_ids; ++i)
            {
                IndexType file_id = ids[i];
                auto *src = cache + file_id * num_bytes;
                auto *dst = reinterpret_cast<char *>(values.get() + value_offset + i * aligned_bytes);
                memcpy(dst, src, num_bytes);
            }
        }

        void save(std::ofstream &writer)
        {
            size_t num_bytes = params.num_bytes();
            constexpr size_t BLOCK_SIZE = 1000000;
            size_t index = 0;

            while (index < num_vectors)
            {
                size_t floor = index;
                size_t ceiling = (index + BLOCK_SIZE <= num_vectors) ? (index + BLOCK_SIZE) : num_vectors;
                size_t batch_count = ceiling - floor;
                byte *data_start = new byte[batch_count * num_bytes];

#pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i)
                {
                    std::memmove(data_start + (i - floor) * num_bytes, values.get() + i * aligned_bytes, num_bytes);
                }

                writer.write(reinterpret_cast<char *>(data_start), batch_count * num_bytes);
                delete[] data_start;
                index = ceiling;
            }
        }

        void resize(size_t new_size)
        {
            if (new_size <= capacity_size)
            {
                num_vectors = new_size;
            }
            else
            {
                std::cout << "Point resize error!" << std::endl;
                abort();
            }
        }

        size_t size() const { return num_vectors; }

        Point operator[](size_t i) const
        {
            if (i >= num_vectors)
            {
                std::cout << "ERROR: Point index out of range: " << i << " from range [0, " << num_vectors << ")" << std::endl;
                abort();
            }
            return Point(values.get() + i * aligned_bytes, params);
        }

        Parameters params;

    protected:
        std::shared_ptr<byte[]> values;
        size_t aligned_bytes;  // 单个向量的对齐后字节数
        size_t num_vectors;    // 当前向量数量
        size_t capacity_size;  // 已分配容量
    };

    // 通过 id 索引间接访问底层 PointRange 的视图
    template <class PointRangeType, typename IndexType>
    struct RefPointRange
    {
        using Point = typename PointRangeType::Point;
        using Parameters = typename PointRangeType::Parameters;
        using T = typename Point::T;

        uint32_t dimension() const { return params.dims; }

        RefPointRange(const IndexType *ids, size_t ids_num, const PointRangeType *raw_points)
            : ids(ids), ids_num(ids_num), raw_points(raw_points), params(raw_points->params)
        {
        }

        size_t size() const { return ids_num; }

        void resize(size_t new_size) { ids_num = new_size; }

        Point operator[](size_t i) const { return (*raw_points)[ids[i]]; }

        Parameters params;

    protected:
        const IndexType *ids;
        size_t ids_num;
        const PointRangeType *raw_points;
    };
}
