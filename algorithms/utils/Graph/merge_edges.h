#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <omp.h>
#include <chrono>
#include "query_params.h"

namespace ant
{

    // 将相同起点的边归并到一起
    template <typename indexType>
    struct MergeEdges
    {
        using IDist = IdDist<indexType>;
        MergeEdges(size_t max_edge_num, uint32_t max_visited_size, uint32_t num_workers, uint32_t max_degree) : max_visited_size(max_visited_size),
                                                                                                                all_edges(max_edge_num),
                                                                                                                nextPos(max_edge_num),
                                                                                                                merge_edges_view((max_visited_size + max_degree) * (size_t)num_workers),
                                                                                                                num_workers(num_workers),
                                                                                                                max_degree(max_degree)
        {
            clear();
        }

        void clear()
        {
            all_edges.clear();
            num_merge = 0;
        }

        void insert_edge(indexType i, IDist j)
        {
            all_edges.push_back(std::make_pair(i, j));
        }

        /*void merge_edges()
        {
            if (all_edges.size() == 0)
                return;

            // 利用排序归并同一起点的边
            std::sort(all_edges.begin(), all_edges.end(), [](std::pair<indexType, PID> a, std::pair<indexType, PID> b)
                      {
                          if (a.first < b.first)
                              return true;
                          if (a.first == b.first)
                          {
                              if (a.second.second < b.second.second)
                              {
                                  return true;
                              }
                              if (a.second.second == b.second.second)
                              {
                                  return a.second.first < b.second.first;
                              }
                          }
                          return false;
                      });

            // 记录每个起点的邻居数，以及起点id

            size_t cur_merge_edges_num = 0;
            for (size_t i = 0; i < all_edges.size() + 1; ++i)
            {
                if (i == all_edges.size() || (i > 0 && all_edges[i].first != all_edges[i - 1].first))
                {
                    // 遇到新的边，需要保存之前的边
                    size_t pre_pos_num = (num_merge == 0) ? 0 : nextPos[num_merge - 1];
                    nextPos[num_merge] = cur_merge_edges_num + pre_pos_num;
                    num_merge++;
                    cur_merge_edges_num = 0;
                    if (i == all_edges.size())
                        break;
                }
                ++cur_merge_edges_num;
            }
        }*/

        using PairType = std::pair<indexType, IDist>;

        // 并行快速排序（OpenMP任务版本）
        void parallelQuickSort(typename std::vector<PairType>::iterator begin,
                               typename std::vector<PairType>::iterator end,
                               int depth = 0)
        {
            if (begin >= end)
                return;

            // 规模阈值：小于10000条边使用串行排序
            const size_t CUTOFF = 10000;

            if (std::distance(begin, end) < static_cast<long>(CUTOFF) || depth > 8)
            {
                std::sort(begin, end, [](const PairType &a, const PairType &b)
                          {
                          if (a.first < b.first)
                              return true;
                          if (a.first == b.first)
                          {
                              return id_dist_less(a.second, b.second);
                          }
                          return false; });
                return;
            }

            // 选择中间元素作为枢轴
            auto mid = begin + std::distance(begin, end) / 2;
            std::iter_swap(begin, mid);
            const auto &pivot = *begin;

            // 分区操作
            auto it = std::partition(begin + 1, end,
                                     [&pivot](const PairType &e)
                                     {
                                         if (e.first < pivot.first)
                                         {
                                             return true;
                                         }
                                         if (e.first == pivot.first)
                                         {
                                             return id_dist_less(e.second, pivot.second);
                                         }
                                         return false;
                                     });
            std::iter_swap(begin, it - 1);

// 递归并行处理两个分区
#pragma omp task if (depth < 6)
            parallelQuickSort(begin, it - 1, depth + 1);

#pragma omp task if (depth < 6)
            parallelQuickSort(it, end, depth + 1);
        }

        void merge_edges()
        {
            if (all_edges.size() == 0)
                return;

#pragma omp parallel
            {
#pragma omp single nowait
                parallelQuickSort(all_edges.begin(), all_edges.end());
            }

            // 记录每个起点的邻居数，以及起点id

            size_t cur_merge_edges_num = 0;
            for (size_t i = 0; i < all_edges.size() + 1; ++i)
            {
                if (i == all_edges.size() || (i > 0 && all_edges[i].first != all_edges[i - 1].first))
                {
                    // 遇到新的边，需要保存之前的边
                    size_t pre_pos_num = (num_merge == 0) ? 0 : nextPos[num_merge - 1];
                    nextPos[num_merge] = cur_merge_edges_num + pre_pos_num;
                    num_merge++;
                    cur_merge_edges_num = 0;
                    if (i == all_edges.size())
                        break;
                }
                ++cur_merge_edges_num;
            }
        }

        indexType get_merge_start_index(size_t merge_id)
        {
            return all_edges[nextPos[merge_id] - 1].first;
        }

        std::pair<IDist *, size_t> get_merge_end_indices(size_t merge_id, uint32_t worker_id)
        {
            // 先读取到缓存
            IDist *cur_cache = merge_edges_view.data() + worker_id * (max_visited_size + max_degree);
            size_t pre_pos = (merge_id == 0) ? 0 : nextPos[merge_id - 1];
            size_t end_pos = std::min(nextPos[merge_id], pre_pos + max_visited_size);
            for (auto j = pre_pos; j < end_pos; ++j)
            {
                cur_cache[j - pre_pos] = all_edges[j].second;
            }
            return std::make_pair(cur_cache, end_pos - pre_pos);
        }

        size_t get_num_merge()
        {
            return num_merge;
        }

    protected:
        // 存储反转前的边
        std::vector<std::pair<indexType, IDist>> all_edges;
        // size_t num_edges;

        // 记录某个边在数组中的结束位置，注意会预留max_degree个位置
        std::vector<size_t> nextPos;
        size_t num_merge;

        // 读取合并后的数据时，需要搬运到这里
        std::vector<IDist> merge_edges_view;
        const uint32_t num_workers;
        const uint32_t max_degree;
        const uint32_t max_visited_size;
    };

}