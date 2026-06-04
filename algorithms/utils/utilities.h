#pragma once
#include <cstdint>
#include <vector>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <cstring>
#include <functional>
#include <random>
#include <filesystem>

namespace ant
{
    // from numerical recipes
    inline uint64_t hash64(uint64_t u)
    {
        uint64_t v = u * 3935559000370003845ul + 2691343689449507681ul;
        v ^= v >> 21;
        v ^= v << 37;
        v ^= v >> 4;
        v *= 4768777513237032717ul;
        v ^= v << 20;
        v ^= v >> 41;
        v ^= v << 5;
        return v;
    }

    // a slightly cheaper, but possibly not as good version
    // based on splitmix64
    inline uint64_t hash64_2(uint64_t x)
    {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        x = x ^ (x >> 31);
        return x;
    }

    struct Random
    {
    public:
        Random(size_t seed) : state(seed) {}
        Random() : state(0) {}
        Random fork(uint64_t i) const { return Random(static_cast<size_t>(hash64(hash64(i + state)))); }
        Random next() const { return fork(0); }
        size_t ith_rand(uint64_t i) const { return static_cast<size_t>(hash64(i + state)); }
        size_t operator[](size_t i) const { return ith_rand(i); }
        size_t rand() { return ith_rand(0); }
        size_t max() { return std::numeric_limits<size_t>::max(); }

    private:
        size_t state = 0;
    };

    template<typename indexType>
    int randomInt(indexType n)
    {
        // 静态随机数引擎（线程局部版本，避免重复初始化开销）
        static thread_local std::mt19937 engine(std::random_device{}());
        // 均匀分布对象
        std::uniform_int_distribution<indexType> dist(0, n - 1);
        return dist(engine);
    }

    template <typename dataType>
    void flat(std::vector<std::vector<dataType>> &a, std::vector<dataType> &flat_a)
    {
        // 预先分配空间（可选，提升性能）
        std::vector<size_t> offset;
        size_t total_size = 0;
        for (const auto &inner : a)
        {
            offset.push_back(total_size);
            total_size += inner.size();
        }
        flat_a.resize(total_size);

// 打平
#pragma omp parallel for
        for (size_t i = 0; i < a.size(); ++i)
        {
            auto &inner = a[i];
            memcpy(flat_a.data() + offset[i], inner.data(), inner.size() * sizeof(dataType));
        }
    }

    template <typename indexType, typename dataType>
    void groupby(std::vector<std::pair<indexType, dataType>> &all_edges, std::vector<std::pair<indexType, std::vector<dataType>>> &all_groupby_res)
    {
        /*
        // 步骤1: 使用 unordered_map 累积分组数据
        std::unordered_map<indexType, std::vector<dataType>> temp_map;
        temp_map.reserve(all_edges.size()); // 预分配空间避免重新哈希

        for (const auto& e : all_edges) {
            // 高效地将 second 添加到对应 first 的 vector 中
            temp_map[e.first].push_back(e.second);
        }

        // 步骤2: 转换为目标数据结构
        all_groupby_res.reserve(temp_map.size());
        for (auto& pair : temp_map) {
            // 使用 move 避免拷贝，提升性能
            all_groupby_res.emplace_back(pair.first, std::move(pair.second));
        }*/
        std::sort(all_edges.begin(), all_edges.end(), [](auto a, auto b)
                  { return a.first < b.first; });

        all_groupby_res.resize(0);
        indexType pre_index = -1;
        std::vector<dataType> *cur_res;
        for (auto edge : all_edges)
        {
            if (edge.first != pre_index)
            {
                pre_index = edge.first;
                std::vector<dataType> res = {edge.second};
                all_groupby_res.push_back(std::pair<indexType, std::vector<dataType>>(pre_index, res));
                cur_res = &all_groupby_res[all_groupby_res.size() - 1].second;
            }
            else
            {
                cur_res->push_back(edge.second);
            }
        }
    }

    template <typename T>
    size_t merge_union(const T *arr1, size_t size1,
                       const T *arr2, size_t size2,
                       T *result, size_t n, std::function<bool(const T &, const T &)> less)
    {
        size_t i = 0, j = 0, k = 0;

        while (i < size1 && j < size2 && k < n)
        {
            if (less(arr1[i], arr2[j]))
            {
                result[k++] = arr1[i++];
            }
            else if (less(arr2[j], arr1[i]))
            {
                result[k++] = arr2[j++];
            }
            else
            {
                // 相等，只取一个
                result[k++] = arr1[i++];
                j++;
            }
        }

        // 处理剩余元素
        while (i < size1 && k < n)
        {
            result[k++] = arr1[i++];
        }

        while (j < size2 && k < n)
        {
            result[k++] = arr2[j++];
        }

        return k; // 返回实际写入的元素数量
    }

    bool ends_with(const std::string &str, const std::string &suffix)
    {
        if (suffix.size() > str.size())
            return false;
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    bool file_exists(const std::string &filename)
    {
        return std::filesystem::exists(filename);
    }
}