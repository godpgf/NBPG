#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>
#include <type_traits>

namespace ant
{
    // 每个字节打包 2 个 4-bit 无符号量化值（.u4bin 格式）
    enum class uint4_2 : uint8_t
    {
    };

    // 每个字节打包 2 个 4-bit 有符号量化值（.i4bin 格式）
    enum class int4_2 : int8_t
    {
    };

    enum class DistanceMetric : uint8_t
    {
        EUCLIDEAN, // 欧氏距离（返回平方距离，便于比较）
        MIPS,      // 最大内积搜索（返回负内积）
    };

    // 从打包字节中解出高/低半字节分量（与 kmeans.h::depress 约定一致）
    inline void unpack_packed_byte(uint8_t byte, int32_t &hi, int32_t &lo)
    {
        constexpr uint8_t HIGH_NIBBLE_MASK = 0xF0;
        hi = static_cast<int32_t>(byte & HIGH_NIBBLE_MASK);
        lo = static_cast<int32_t>(static_cast<uint8_t>(byte << 4));
    }

    // 计算平方欧氏距离；p、q 为连续存储的 d 维向量
    float euclidean_distance(const uint8_t *p, const uint8_t *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t diff = static_cast<int32_t>(p[i]) - static_cast<int32_t>(q[i]);
            result += diff * diff;
        }
        return static_cast<float>(result);
    }

    float mips_distance(const uint8_t *p, const uint8_t *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            result += static_cast<int32_t>(p[i]) * static_cast<int32_t>(q[i]);
        }
        return -static_cast<float>(result);
    }

    float euclidean_distance(const int8_t *p, const int8_t *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t diff = static_cast<int32_t>(p[i]) - static_cast<int32_t>(q[i]);
            result += diff * diff;
        }
        return static_cast<float>(result);
    }

    float mips_distance(const int8_t *p, const int8_t *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            result += static_cast<int32_t>(p[i]) * static_cast<int32_t>(q[i]);
        }
        return -static_cast<float>(result);
    }

    float euclidean_distance(const uint4_2 *p, const uint4_2 *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t p_hi, p_lo, q_hi, q_lo;
            unpack_packed_byte(static_cast<uint8_t>(p[i]), p_hi, p_lo);
            unpack_packed_byte(static_cast<uint8_t>(q[i]), q_hi, q_lo);

            result += (p_hi - q_hi) * (p_hi - q_hi);
            result += (p_lo - q_lo) * (p_lo - q_lo);
        }
        return static_cast<float>(result);
    }

    float mips_distance(const uint4_2 *p, const uint4_2 *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t p_hi, p_lo, q_hi, q_lo;
            unpack_packed_byte(static_cast<uint8_t>(p[i]), p_hi, p_lo);
            unpack_packed_byte(static_cast<uint8_t>(q[i]), q_hi, q_lo);

            result += p_hi * q_hi;
            result += p_lo * q_lo;
        }
        return -static_cast<float>(result);
    }

    float euclidean_distance(const int4_2 *p, const int4_2 *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t p_hi, p_lo, q_hi, q_lo;
            unpack_packed_byte(static_cast<uint8_t>(p[i]), p_hi, p_lo);
            unpack_packed_byte(static_cast<uint8_t>(q[i]), q_hi, q_lo);

            result += (p_hi - q_hi) * (p_hi - q_hi);
            result += (p_lo - q_lo) * (p_lo - q_lo);
        }
        return static_cast<float>(result);
    }

    float mips_distance(const int4_2 *p, const int4_2 *q, unsigned dims)
    {
        int32_t result = 0;
        for (unsigned i = 0; i < dims; ++i)
        {
            int32_t p_hi, p_lo, q_hi, q_lo;
            unpack_packed_byte(static_cast<uint8_t>(p[i]), p_hi, p_lo);
            unpack_packed_byte(static_cast<uint8_t>(q[i]), q_hi, q_lo);

            result += p_hi * q_hi;
            result += p_lo * q_lo;
        }
        return -static_cast<float>(result);
    }

    float euclidean_distance(const float *p, const float *q, unsigned dims)
    {
        float result = 0.0f;
        for (unsigned i = 0; i < dims; ++i)
        {
            float diff = p[i] - q[i];
            result += diff * diff;
        }
        return result;
    }

    float mips_distance(const float *p, const float *q, unsigned dims)
    {
        float result = 0.0f;
        for (unsigned i = 0; i < dims; ++i)
        {
            result += p[i] * q[i];
        }
        return -result;
    }

    // 单个向量视图，不持有内存，仅通过指针访问外部存储
    template <typename T_>
    struct Point
    {
        using T = T_;
        using byte = uint8_t;

        struct Parameters
        {
            unsigned dims;                    // 逻辑维度（打包类型下为字节数）
            DistanceMetric distance_metric;   // 距离度量
            size_t num_bytes() const { return dims * sizeof(T); }
            Parameters() : dims(0), distance_metric(DistanceMetric::EUCLIDEAN) {}
            Parameters(unsigned dims, DistanceMetric distance_metric = DistanceMetric::EUCLIDEAN)
                : dims(dims), distance_metric(distance_metric) {}
            Parameters(const Parameters &p) : dims(p.dims), distance_metric(p.distance_metric) {}
        };

        // 从当前向量中截取子向量视图，offset 为元素偏移，sub_dims 为子向量维度
        Point get_sub_point(unsigned offset, unsigned sub_dims) const
        {
            return Point(reinterpret_cast<byte *>(values + offset), Parameters(sub_dims));
        }

        T operator[](unsigned i) const { return values[i]; }
        T &operator[](unsigned i) { return values[i]; }
        T *data() const { return values; }

        // 按缓存行预取向量数据
        void prefetch() const
        {
            auto cache_lines = (params.num_bytes() - 1) / 64 + 1;
            for (auto i = 0; i < cache_lines; ++i)
            {
                __builtin_prefetch(reinterpret_cast<char *>(values) + i * 64);
            }
        }

        Point() : values(nullptr), params() {}

        Point(byte *values, const Parameters &params)
            : values(reinterpret_cast<T *>(values)), params(params) {}

        // 仅支持 float 向量的随机初始化
        void rand_init(std::mt19937 &gen, std::uniform_real_distribution<float> &dist)
        {
            if constexpr (std::is_same_v<T_, float>)
            {
                for (unsigned j = 0; j < params.dims; ++j)
                {
                    values[j] = dist(gen);
                }
            }
            else
            {
                static_assert(std::is_same_v<T_, float>, "rand_init only supports float vectors");
            }
        }

        // 判断两个 Point 是否指向同一块内存
        bool same_as(const Point &other) const { return values == other.values; }

        float dot(const Point<T_> &other) const
        {
            return -mips_distance(data(), other.data(), params.dims);
        }

        void add_feat(std::pair<unsigned, T_> *features, float weight = 1.0f)
        {
            add_feat(features, params.dims, weight);
        }

        // 稀疏特征累加：features[i] = {维度下标, 特征值}
        void add_feat(std::pair<unsigned, T_> *features, unsigned sparse_dims, float weight)
        {
            for (unsigned i = 0; i < sparse_dims; ++i)
            {
                values[features[i].first] += features[i].second * weight;
            }
        }

        void add_feat(const Point<T_> &other, float weight)
        {
            for (unsigned i = 0; i < params.dims; ++i)
            {
                values[i] += other.values[i] * weight;
            }
        }

        void add_feat_square(const Point<T_> &other, float weight)
        {
            for (unsigned i = 0; i < params.dims; ++i)
            {
                values[i] += other.values[i] * other.values[i] * weight;
            }
        }

        Point &operator+=(const Point<T_> &other)
        {
            for (unsigned i = 0; i < params.dims; ++i)
            {
                values[i] += other.values[i];
            }
            return *this;
        }

        Point &operator/=(float scale)
        {
            for (unsigned i = 0; i < params.dims; ++i)
            {
                values[i] /= scale;
            }
            return *this;
        }

        Point &operator*=(float scale)
        {
            for (unsigned i = 0; i < params.dims; ++i)
            {
                values[i] *= scale;
            }
            return *this;
        }

        // 均方根；eps 防止除零
        float rms(float eps = 1e-6f) const
        {
            float sum_sq = 0.0f;
            for (unsigned i = 0; i < params.dims; ++i)
            {
                sum_sq += values[i] * values[i];
            }
            return std::sqrt(sum_sq / params.dims + eps);
        }

        float norm() const
        {
            float sum_sq = 0.0f;
            for (unsigned i = 0; i < params.dims; ++i)
            {
                sum_sq += values[i] * values[i];
            }
            return std::sqrt(sum_sq);
        }

        float distance(const Point<T_> &other) const
        {
            switch (params.distance_metric)
            {
            case DistanceMetric::EUCLIDEAN:
                return euclidean_distance(values, other.values, params.dims);
            case DistanceMetric::MIPS:
                return mips_distance(values, other.values, params.dims);
            default:
                return 0;
            }
        }

        void copy_(const Point<T_> &other)
        {
            std::memcpy(values, other.values, params.dims * sizeof(T_));
        }

        void zeros_()
        {
            std::memset(values, 0, params.dims * sizeof(T_));
        }

        Parameters params;

    protected:
        T *values;
    };

}
