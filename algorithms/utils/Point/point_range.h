#pragma once
#include <cstdint>
#include <sys/mman.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <memory>
#include <cstring>
#include <random>
#include <algorithm>
#include "point.h"

namespace ant
{

    template<typename indexType, typename PR>
    PR loadPointRange(const char* filename, DistanceMetric distance_metric=DistanceMetric::EUCLIDIAN){
        std::ifstream reader(filename, std::ios::in | std::ios::binary);
        assert(reader.is_open());
        indexType n, dims;
        using Parameters = typename PR::Parameters;
        reader.read((char*)&n, sizeof(indexType));
        reader.read((char*)&dims, sizeof(indexType));
        auto Points = PR(n, Parameters(dims, distance_metric));
        Points.load(reader);
        reader.close();
        return std::move(Points);
    }

    template <class Point_>
    struct PointRange
    {
        using Point = Point_;
        using Parameters = typename Point::Parameters;
        using byte = uint8_t;
        using T = typename Point::T;

        unsigned dimension() const { return params.dims; }
        size_t capacity() { return capacity_size; }

        PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)), capacity_size(0) {}

        PointRange(const PointRange&) = default;

        PointRange(size_t num_vectors, const Parameters p) : n(num_vectors), capacity_size(num_vectors), params(p)
        {
            size_t num_bytes = params.num_bytes();
            aligned_bytes = 64 * ((num_bytes - 1) / 64 + 1);
            size_t total_bytes = capacity_size * aligned_bytes;
            byte *ptr = (byte *)aligned_alloc(1l << 21, total_bytes);
            madvise(ptr, total_bytes, MADV_HUGEPAGE);
            values = std::shared_ptr<byte[]>(ptr, std::free);
        }

        // 初始化和清除缓存
        void init_cache(){}
        void clear_cache(){}

        template<typename OtherPoint>
        PointRange<Point_> &operator+=(const PointRange<OtherPoint> &x) {
            #pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i) {
                (*this)[i] += x[i];
            }
            return *this;
        }

        PointRange<Point_> &operator*=(const float w) {
            #pragma omp parallel for
            for (auto i = 0; i < this->size(); ++i) {
                (*this)[i] *= w;
            }
            return *this;
        }

        PointRange<Point_> &operator=(const PointRange<Point_> &x) {
            this->ref_copy_(x);
            return *this;
        }

        PointRange<Point_> &operator=(PointRange<Point_> &&x) noexcept {
            this->ref_copy_(x);
            return *this;
        }

        template<typename OtherPoint>
        void add_feat(PointRange<OtherPoint> &x, float weight) {
            assert(x.size() == this->size());
            #pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i) {
                auto p = x[i];
                (*this)[i].add_feat(p, weight);
            }
        }

        template<typename OtherPoint>
        void add_feat_square(PointRange<OtherPoint> &x, float weight) {
            assert(x.size() == this->size());
            #pragma omp parallel for
            for (auto i = 0; i < x.size(); ++i) {
                auto p = x[i];
                (*this)[i].add_feat_square(p, weight);
            }
        }

        void copy_(const PointRange<Point_> &x, size_t from_id = 0, size_t to_id = 0, size_t copy_size = static_cast<size_t>(-1)) {
            if (copy_size == static_cast<size_t>(-1)) {
                copy_size = n - to_id;
            }
            while (copy_size > 0) {
                memcpy((*this)[to_id++].data(), x[from_id++].data(), params.num_bytes());
                copy_size--;
            }
        }

        void ref_copy_(PointRange<Point_>& other) {
            params = other.params;
            aligned_bytes = other.aligned_bytes;
            capacity_size = other.capacity_size;
            n = other.n;
            values = other.values;
        }

        void zeros_() {
            memset(values.get(), 0, aligned_bytes * n);
        }

        void kaiming_uniform_init(unsigned fan, float a = 0) {
            float bound = std::sqrt(6.0f / ((1.0f + a * a) * fan));

            std::random_device rd;

            #pragma omp parallel
            {
                std::mt19937 gen(rd() + omp_get_thread_num());
                std::uniform_real_distribution<float> dist(-bound, bound);

                #pragma omp for
                for (size_t i = 0; i < n; ++i) {
                    (*this)[i].rand_init(gen, dist);
                }
            }
        }

        void load(std::ifstream &reader, size_t start_id = 0) {
            size_t BLOCK_SIZE = 1000000;
            size_t index = 0;
            size_t num_bytes = params.num_bytes();
            std::vector<byte> block_data(BLOCK_SIZE * num_bytes);

            while (start_id + index < n) {
                size_t floor = index;
                size_t ceiling = (index + BLOCK_SIZE <= n) ? (index + BLOCK_SIZE) : n;
                long m = static_cast<long>(ceiling - floor);
                byte *data_start = block_data.data();
                reader.read((char *)(data_start), m * num_bytes);

                #pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i) {
                    std::memmove(values.get() + (i + start_id) * aligned_bytes,
                                 data_start + (i - floor) * num_bytes,
                                 num_bytes);
                }
                index = ceiling;
            }
        }

        template<typename indexType>
        void load(const char* filename, size_t headOffset, const indexType* ids, size_t num_ids, size_t start_id = 0) {
            std::ifstream reader(filename, std::ios::in | std::ios::binary);
            load<indexType>(reader, headOffset, ids, num_ids, start_id);
            reader.close();
        }

        template<typename indexType>
        void load(std::ifstream &reader, size_t headOffset, const indexType* ids, size_t num_ids, size_t start_id = 0, size_t max_batch_size = 100000) {
            assert(reader.is_open());
            size_t value_offset = start_id * aligned_bytes;
            size_t num_bytes = params.num_bytes();

            auto max_it = std::max_element(ids, ids + num_ids);
            size_t max_n = (*max_it) + 1;
            std::vector<indexType> id2offset(max_n, static_cast<indexType>(-1));

            assert(num_ids + start_id <= n);
            for (size_t i = 0; i < num_ids; ++i) {
                indexType cur_id = ids[i];
                id2offset[cur_id] = static_cast<indexType>(i);
            }

            std::vector<char> cache(max_batch_size * num_bytes);
            reader.seekg(headOffset);

            for (size_t i = 0; i < max_n; i += max_batch_size) {
                size_t cur_batch_size = (i + max_batch_size) > max_n ? (max_n - i) : max_batch_size;
                reader.read(cache.data(), cur_batch_size * num_bytes);

                for (size_t j = 0; j < cur_batch_size; ++j) {
                    size_t index = i + j;
                    if (id2offset[index] < 0 || id2offset[index] >= static_cast<indexType>(max_n))
                        continue;

                    auto* ptr = cache.data() + j * num_bytes;
                    auto* to_ptr = (char*)(values.get() + value_offset + id2offset[index] * aligned_bytes);
                    memcpy(to_ptr, ptr, num_bytes);
                }
            }
        }

        template<typename indexType>
        void load(char* cache, const indexType* ids, size_t num_ids, size_t start_id = 0, size_t max_batch_size = 100000) {
            size_t value_offset = start_id * aligned_bytes;
            size_t num_bytes = params.num_bytes();

            #pragma omp parallel for
            for (size_t i = 0; i < num_ids; ++i) {
                indexType index = ids[i];
                auto* ptr = cache + index * num_bytes;
                auto* to_ptr = (char*)(values.get() + value_offset + i * aligned_bytes);
                memcpy(to_ptr, ptr, num_bytes);
            }
        }

        void save(std::ofstream &writer) {
            size_t num_bytes = params.num_bytes();
            size_t BLOCK_SIZE = 1000000;
            size_t index = 0;

            while (index < n) {
                size_t floor = index;
                size_t ceiling = (index + BLOCK_SIZE <= n) ? (index + BLOCK_SIZE) : n;
                size_t m = ceiling - floor;
                byte *data_start = new byte[m * num_bytes];

                #pragma omp parallel for
                for (size_t i = floor; i < ceiling; ++i) {
                    std::memmove(data_start + (i - floor) * num_bytes, values.get() + i * aligned_bytes, num_bytes);
                }

                writer.write((char *)(data_start), m * num_bytes);
                delete[] data_start;
                index = ceiling;
            }
        }

        void resize(size_t new_size) {
            if (new_size <= capacity_size) {
                n = new_size;
            } else {
                std::cout << "Point resize error!" << std::endl;
                abort();
            }
        }

        size_t size() const {
            return n;
        }

        Point operator[](size_t i) const {
            if (i >= n) {
                std::cout << "ERROR: Point index out of range: " << i << " from range [0, " << n << ")" << std::endl;
                abort();
            }
            return Point(values.get() + i * aligned_bytes, params);
        }

        Parameters params;

    protected:
        std::shared_ptr<byte[]> values;
        size_t aligned_bytes;
        size_t n;
        size_t capacity_size;
    };

    template <class PR, typename indexType>
    struct RefPointRange
    {
        using Point = typename PR::Point;
        using parameters = typename Point::parameters;
        using T = typename Point::T;
        uint32_t dimension() const { return params.dims; }

        RefPointRange(const indexType *ids, size_t ids_num, const PR *raw_points) : ids(ids), ids_num(ids_num), raw_points(raw_points), params(raw_points->params)
        {
        }

        size_t size() const { 
            return ids_num;
        }

        void resize(size_t new_size){
            ids_num = new_size;
        }

        Point operator[](size_t i) const
        {
            return (*raw_points)[ids[i]];
        }

        parameters params;

    protected:
        const indexType *ids;
        size_t ids_num;
        const PR *raw_points;
    };
}