#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include "query_params.h"

namespace ant
{

    // -------------------------------------------------------------------------
    // MergeEdges — 收集反向边并按 target 分组，供 batch_update_graph 并行写回
    //
    // insert_edge(target, {source, dist})：A 连 B 时插入反向边 B→A。
    // build_groups() / merge_edges()：排序 all_edges，填充 nextPos（每组在 all_edges 中的独占结束下标）。
    // group_neighbors()：把一组的 IdDist（source + dist）拷入 worker 私有缓冲区。
    // -------------------------------------------------------------------------
    template <typename indexType>
    struct MergeEdges
    {
        using IDist = IdDist<indexType>;
        using Edge = std::pair<indexType, IDist>;

        MergeEdges(size_t max_edge_num, uint32_t max_visited_size, uint32_t num_workers_in, uint32_t max_degree)
            : max_visited_size(max_visited_size),
              max_degree(max_degree),
              num_workers(num_workers_in == 0 ? 1u : num_workers_in),
              worker_buffer_stride(max_visited_size + max_degree),
              num_merge(0),
              merge_edges_view(static_cast<size_t>(num_workers) * worker_buffer_stride)
        {
            all_edges.reserve(max_edge_num);
            nextPos.reserve(max_edge_num);
            clear();
        }

        void clear()
        {
            all_edges.clear();
            num_merge = 0;
        }

        void insert_edge(indexType target, IDist neighbor)
        {
            all_edges.emplace_back(target, neighbor);
        }

        // 排序并建立分组边界 nextPos[0..num_merge)
        void merge_edges()
        {
            if (all_edges.empty())
            {
                num_merge = 0;
                return;
            }

            std::sort(all_edges.begin(), all_edges.end(), edge_less);

            num_merge = 0;
            if (nextPos.size() < all_edges.size() + 1)
                nextPos.resize(all_edges.size() + 1);

            size_t cur_group_size = 0;
            for (size_t i = 0; i < all_edges.size() + 1; ++i)
            {
                if (i == all_edges.size() || (i > 0 && all_edges[i].first != all_edges[i - 1].first))
                {
                    const size_t prev_end = (num_merge == 0) ? 0 : nextPos[num_merge - 1];
                    nextPos[num_merge] = cur_group_size + prev_end;
                    ++num_merge;
                    cur_group_size = 0;
                    if (i == all_edges.size())
                        break;
                }
                ++cur_group_size;
            }
        }

        void build_groups() { merge_edges(); }

        size_t get_num_merge() const { return num_merge; }

        size_t num_groups() const { return num_merge; }

        indexType get_merge_start_index(size_t merge_id) const
        {
            assert(merge_id < num_merge);
            assert(nextPos[merge_id] > 0 && nextPos[merge_id] <= all_edges.size());
            return all_edges[nextPos[merge_id] - 1].first;
        }

        indexType group_source(size_t group_id) const { return get_merge_start_index(group_id); }

        std::pair<IDist *, size_t> get_merge_end_indices(size_t merge_id, uint32_t worker_id)
        {
            assert(merge_id < num_merge);
            assert(!all_edges.empty());

            const size_t slot = static_cast<size_t>(worker_id % num_workers);
            IDist *buf = merge_edges_view.data() + slot * worker_buffer_stride;

            const size_t begin = (merge_id == 0) ? 0 : nextPos[merge_id - 1];
            const size_t group_end = nextPos[merge_id];
            assert(begin <= group_end && group_end <= all_edges.size());

            const size_t end = std::min(group_end, begin + static_cast<size_t>(max_visited_size));
            assert(end <= all_edges.size());
            assert(end - begin <= worker_buffer_stride);

            for (size_t j = begin; j < end; ++j)
            {
                buf[j - begin] = all_edges[j].second;
            }
            return {buf, end - begin};
        }

        std::pair<IDist *, size_t> group_neighbors(size_t group_id, uint32_t worker_id)
        {
            return get_merge_end_indices(group_id, worker_id);
        }

        uint32_t buffer_capacity() const { return worker_buffer_stride; }

    private:
        static bool edge_less(const Edge &a, const Edge &b)
        {
            if (a.first != b.first)
                return a.first < b.first;
            return id_dist_less(a.second, b.second);
        }

        std::vector<Edge> all_edges;
        // nextPos[k] = 第 k 组在 all_edges 中的独占结束下标（与原版一致）
        std::vector<size_t> nextPos;
        size_t num_merge;

        // 配置字段必须排在 merge_edges_view 之前，保证初始化顺序正确
        const uint32_t max_visited_size;
        const uint32_t max_degree;
        const uint32_t num_workers;
        const uint32_t worker_buffer_stride;

        std::vector<IDist> merge_edges_view;
    };

} // namespace ant
