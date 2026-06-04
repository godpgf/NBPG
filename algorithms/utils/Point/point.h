#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>

namespace ant
{
    enum class uint4_2 : uint8_t
    {
    };
    enum class int4_2 : int8_t
    {
    };

    enum class DistanceMetric : uint8_t
    {
        EUCLIDIAN,
        MIPS,
    };

    float euclidian_distance(const uint8_t *p, const uint8_t *q, unsigned d)
    {
        int32_t result = 0;
        for (int i = 0; i < d; i++)
        {
            int32_t diff = (int32_t)p[i] - (int32_t)q[i];
            result += diff * diff;
        }
        return (float)result;
    }

    float mips_distance(const uint8_t *p, const uint8_t *q, unsigned d)
    {
        int32_t result = 0;
        for (int i = 0; i < d; i++)
        {
            result += (int32_t)p[i] * (int32_t)q[i];
        }
        return - (float)result;
    }

    float euclidian_distance(const int8_t *p, const int8_t *q, unsigned d)
    {
        int32_t result = 0;
        for (int i = 0; i < d; i++)
        {
            int32_t diff = (int32_t)p[i] - (int32_t)q[i];
            result += diff * diff;
        }
        return (float)result;
    }

    float mips_distance(const int8_t *p, const int8_t *q, unsigned d)
    {
        int32_t result = 0;
        for (int i = 0; i < d; i++)
        {
            result += (int32_t)p[i] * (int32_t)q[i];
        }
        return - (float)result;
    }

    float euclidian_distance(const uint4_2 *p, const uint4_2 *q, unsigned d)
    {
        int32_t result = 0;
        const uint8_t MASK = static_cast<uint8_t>(0xF0);
        for (int i = 0; i < d; i++)
        {
            auto cp = static_cast<uint8_t>(p[i]);
            auto cq = static_cast<uint8_t>(q[i]);
            const int32_t qi = (int32_t)(cp & MASK);
            const int32_t pi = (int32_t)(cq & MASK);
            const int32_t qi_ = static_cast<uint8_t>(cp << 4);
            const int32_t pi_ = static_cast<uint8_t>(cq << 4);

            result += (qi - pi) * (qi - pi);
            result += (qi_ - pi_) * (qi_ - pi_);
        }
        return (float)result;
    }

    float mips_distance(const uint4_2 *p, const uint4_2 *q, unsigned d)
    {
        int32_t result = 0;
        const uint8_t MASK = static_cast<uint8_t>(0xF0);
        for (int i = 0; i < d; i++)
        {
            auto cp = static_cast<uint8_t>(p[i]);
            auto cq = static_cast<uint8_t>(q[i]);
            const int32_t qi = (int32_t)(cp & MASK);
            const int32_t pi = (int32_t)(cq & MASK);
            const int32_t qi_ = static_cast<uint8_t>(cp << 4);
            const int32_t pi_ = static_cast<uint8_t>(cq << 4);

            result += (qi * pi);
            result += (qi_ * pi_);
        }
        return - (float)result;
    }

    float euclidian_distance(const int4_2 *p, const int4_2 *q, unsigned d)
    {
        int32_t result = 0;
        const int8_t MASK = static_cast<int8_t>(0xF0);
        for (int i = 0; i < d; i++)
        {
            auto cp = static_cast<int8_t>(p[i]);
            auto cq = static_cast<int8_t>(q[i]);
            const int32_t qi = static_cast<int8_t>(cp & MASK);
            const int32_t pi = static_cast<int8_t>(cq & MASK);
            const int32_t qi_ = static_cast<int8_t>(cp << 4);
            const int32_t pi_ = static_cast<int8_t>(cq << 4);

            result += (qi - pi) * (qi - pi);
            result += (qi_ - pi_) * (qi_ - pi_);
        }
        return (float)result;
    }

    float mips_distance(const int4_2 *p, const int4_2 *q, unsigned d)
    {
        int32_t result = 0;
        const int8_t MASK = static_cast<int8_t>(0xF0);
        for (int i = 0; i < d; i++)
        {
            auto cp = static_cast<int8_t>(p[i]);
            auto cq = static_cast<int8_t>(q[i]);
            const int32_t qi = static_cast<int8_t>(cp & MASK);
            const int32_t pi = static_cast<int8_t>(cq & MASK);
            const int32_t qi_ = static_cast<int8_t>(cp << 4);
            const int32_t pi_ = static_cast<int8_t>(cq << 4);

            result += (qi * pi);
            result += (qi_ * pi_);
        }
        return - (float)result;
    }

    float euclidian_distance(const float *p, const float *q, unsigned d)
    {
        float result = 0.0f;
        for (unsigned i = 0; i < d; ++i)
        {
            float diff = p[i] - q[i];
            result += diff * diff;
        }
        return result;
    }

    float mips_distance(const float *p, const float *q, unsigned d)
    {
        float result = 0.0f;
        for (unsigned i = 0; i < d; ++i)
        {
            result += p[i] * q[i];
        }
        return - (float)result;
    }

    template <typename T_>
    struct Point
    {
        using T = T_;
        using byte = uint8_t;

        struct Parameters
        {
            unsigned dims;
            DistanceMetric distance_metric;
            size_t num_bytes() const { return dims * sizeof(T); }
            Parameters() : dims(0), distance_metric(DistanceMetric::EUCLIDIAN) {}
            Parameters(unsigned dims, DistanceMetric distance_metric=DistanceMetric::EUCLIDIAN) : dims(dims), distance_metric(distance_metric) {}
            Parameters(const Parameters &p) : dims(p.dims), distance_metric(p.distance_metric) {}
        };

        Point getSubPoint(unsigned i, unsigned sub_dim) const {
            return Point((byte *)(values + i * sub_dim), Parameters(sub_dim));
        }

        T operator[](long i) const { return *(values + i); }
        T *data() const { return values; }

        void prefetch() const {
            auto l = (params.num_bytes() - 1) / 64 + 1;
            for (auto i = 0; i < l; i++)
                __builtin_prefetch((char *)values + i * 64);
        }

        Point() : values(nullptr), params(0) {}

        Point(byte *values, const Parameters params)
            : values((T *)values), params(params) {}

        void rand_init(std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
            for (unsigned j = 0; j < params.dims; ++j) {
                values[j] = dist(gen);
            }
        }

        bool same_as(const Point &q) const {
            return values == q.values;
        }

        float dot(const Point<T_> &x) const {
            return -mips_distance(data(), x.data(), params.dims);
        }

        void add_feat(std::pair<unsigned, T_>* x, float weight = 1.0f) {
            add_feat(x, params.dims, weight);
        }

        void add_feat(std::pair<unsigned, T_>* x, unsigned sparse_dims, float weight) {
            for (unsigned i = 0; i < sparse_dims; ++i) {
                values[x[i].first] += x[i].second * weight;
            }
        }

        void add_feat(const Point<T_> &x, float weight) {
            for (unsigned i = 0; i < params.dims; ++i) {
                values[i] += x.values[i] * weight;
            }
        }

        void add_feat_square(const Point<T_> &x, float weight) {
            for (unsigned i = 0; i < params.dims; ++i) {
                values[i] += x.values[i] * x.values[i] * weight;
            }
        }

        Point &operator+=(const Point<T_> &other) {
            for (unsigned i = 0; i < params.dims; ++i) {
                values[i] += other.values[i];
            }
            return *this;
        }

        Point &operator/=(float v) {
            for (unsigned i = 0; i < params.dims; ++i) {
                values[i] /= v;
            }
            return *this;
        }

        Point &operator*=(float v) {
            for (unsigned i = 0; i < params.dims; ++i) {
                values[i] *= v;
            }
            return *this;
        }

        float rms(float eps = 1e-6f) const {
            float avg = 0.0f;
            for (unsigned i = 0; i < params.dims; ++i) {
                avg += values[i] * values[i];
            }
            return std::sqrt(avg / params.dims + eps);
        }

        float norm() const {
            float avg = 0.0f;
            for (unsigned i = 0; i < params.dims; ++i) {
                avg += values[i] * values[i];
            }
            return std::sqrt(avg);
        }

        float distance(const Point<T_> &x) const
        {
            switch (params.distance_metric)
            {
            case DistanceMetric::EUCLIDIAN:
                return euclidian_distance(this->values, x.values, this->params.dims);
            case DistanceMetric::MIPS:
                return mips_distance(this->values, x.values, this->params.dims);
            default:
                return 0;
            }
        }

        void copy_(const Point<T_> &x) {
            std::memcpy(values, x.values, params.dims * sizeof(T_));
        }

        void zeros_() {
            std::memset(values, 0, params.dims * sizeof(T_));
        }

        Parameters params;

    protected:
        T *values;
    };

}