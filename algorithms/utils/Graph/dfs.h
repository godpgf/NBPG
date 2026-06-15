#pragma once
#include <algorithm>
#include <queue>
#include <vector>
#include "graph.h"

namespace ant
{

    // 图遍历与孤立节点修复工具。
    // - 单层图（NSW / Vamana 等）：从入口点出发沿邻接边 BFS，标记可达节点；
    // - HNSW 分层图：额外沿 id_in_pre_level 向下一层传播可达性，仅修复 level=0 的底图节点。
    struct DFS
    {
        // 按度数选取 popular 节点：度数最低的 percent/2 与最高的 percent/2 合并返回。
        template <typename indexType>
        static void get_popular_nodes(std::vector<indexType> &node_ids, Graph<indexType> &G,
                                      std::vector<std::pair<indexType, float>> &tmp_node_popular, float percent)
        {
            tmp_node_popular.resize(G.size());
            for (size_t i = 0; i < tmp_node_popular.size(); ++i)
            {
                tmp_node_popular[i].first = static_cast<indexType>(i);
                tmp_node_popular[i].second = static_cast<float>(G[i].size());
            }
            std::sort(tmp_node_popular.begin(), tmp_node_popular.end(),
                      [](const std::pair<indexType, float> &a, const std::pair<indexType, float> &b)
                      {
                          return (a.second < b.second) || (a.second == b.second && a.first < b.first);
                      });

            size_t top_size = static_cast<size_t>(tmp_node_popular.size() * percent * 0.5);
            size_t last_id = tmp_node_popular.size() - 1;
            for (size_t i = 0; i < top_size; ++i)
            {
                tmp_node_popular[top_size + i] = tmp_node_popular[last_id - i];
            }
            tmp_node_popular.resize(top_size * 2);
            node_ids.resize(top_size * 2);
            for (size_t i = 0; i < tmp_node_popular.size(); ++i)
                node_ids[i] = tmp_node_popular[i].first;
        }

    private:
        // 若 from_id 尚不存在到 to_id 的出边且未满度，则追加一条反向边 from_id -> to_id。
        template <typename indexType>
        static bool add_edge_if_possible(Graph<indexType> &G, size_t from_id, size_t to_id)
        {
            auto from_edges = G[from_id];
            if (from_edges.size() >= G.max_degree())
                return false;
            for (size_t k = 0; k < from_edges.size(); ++k)
            {
                if (static_cast<size_t>(from_edges[k]) == to_id)
                    return true;
            }
            from_edges.append_neighbor(static_cast<indexType>(to_id));
            return true;
        }

        // 尝试修复孤立节点 isolated_id：优先从主连通分量内的邻居补反向边，其次尝试任意邻居或入口点。
        template <typename indexType, typename rangeType>
        static void try_repair_isolated_node(Graph<indexType> &G, size_t isolated_id,
                                             const std::vector<int> &flag, rangeType &enter_node_ids)
        {
            auto edges = G[isolated_id];

            // 1. 优先：孤立节点已有出边指向主分量内的邻居，补反向边即可连通
            for (size_t j = 0; j < edges.size(); ++j)
            {
                size_t nid = static_cast<size_t>(edges[j]);
                if (flag[nid] == 1 && add_edge_if_possible(G, nid, isolated_id))
                    return;
            }

            // 2. 次选：出边邻居均不在主分量，仍尝试向任意有容量的邻居补边
            for (size_t j = 0; j < edges.size(); ++j)
            {
                size_t nid = static_cast<size_t>(edges[j]);
                if (add_edge_if_possible(G, nid, isolated_id))
                    return;
            }

            // 3. 零出度：从入口点向孤立节点连边
            for (auto eid : enter_node_ids)
            {
                size_t entry_id = static_cast<size_t>(eid);
                if (flag[entry_id] == 1 && add_edge_if_possible(G, entry_id, isolated_id))
                    return;
            }

            // 4. 兜底：从任意可达节点向孤立节点连边
            for (size_t r = 0; r < flag.size(); ++r)
            {
                if (flag[r] == 1 && add_edge_if_possible(G, r, isolated_id))
                    return;
            }
        }

        // 单层图 BFS 可达性标记
        template <typename indexType>
        static void mark_reachable_nodes(Graph<indexType> &G, const std::vector<size_t> &seeds,
                                         std::vector<int> &flag, size_t batch_size)
        {
            std::queue<size_t> next_visited;
            for (size_t eid : seeds)
            {
                flag[eid] = 1;
                next_visited.push(eid);
            }

            std::vector<size_t> cur_visited(batch_size);
            while (!next_visited.empty())
            {
                size_t cur_visited_size = 0;
                while (cur_visited_size < batch_size && !next_visited.empty())
                {
                    cur_visited[cur_visited_size++] = next_visited.front();
                    next_visited.pop();
                }

                for (size_t i = 0; i < cur_visited_size; ++i)
                {
                    size_t cur_node_id = cur_visited[i];
                    auto edges = G[cur_node_id];
                    for (size_t j = 0; j < edges.size(); ++j)
                    {
                        size_t nid = static_cast<size_t>(edges[j]);
                        if (flag[nid] == 0)
                        {
                            flag[nid] = 1;
                            next_visited.push(nid);
                        }
                    }
                }
            }
        }

        // HNSW 分层图 BFS：除邻接边外，还沿 id_in_pre_level 向下一层传播可达性
        template <typename indexType, typename HierMapping>
        static void mark_reachable_nodes_hier(Graph<indexType> &G, HierMapping *hidMapping,
                                              const std::vector<size_t> &seeds, std::vector<int> &flag,
                                              size_t batch_size)
        {
            std::queue<size_t> next_visited;
            for (size_t eid : seeds)
            {
                flag[eid] = 1;
                next_visited.push(eid);
            }

            std::vector<size_t> cur_visited(batch_size);
            while (!next_visited.empty())
            {
                size_t cur_visited_size = 0;
                while (cur_visited_size < batch_size && !next_visited.empty())
                {
                    cur_visited[cur_visited_size++] = next_visited.front();
                    next_visited.pop();
                }

                for (size_t i = 0; i < cur_visited_size; ++i)
                {
                    size_t cur_node_id = cur_visited[i];

                    // 沿层级映射下降到下一层（level 0 的 pre_id 为负，不再下降）
                    int64_t lower_id = hidMapping->get_id_in_pre_level(static_cast<int64_t>(cur_node_id));
                    if (lower_id >= 0 && flag[static_cast<size_t>(lower_id)] == 0)
                    {
                        flag[static_cast<size_t>(lower_id)] = 1;
                        next_visited.push(static_cast<size_t>(lower_id));
                    }

                    auto edges = G[cur_node_id];
                    for (size_t j = 0; j < edges.size(); ++j)
                    {
                        size_t nid = static_cast<size_t>(edges[j]);
                        if (flag[nid] == 0)
                        {
                            flag[nid] = 1;
                            next_visited.push(nid);
                        }
                    }
                }
            }
        }

    public:
        // 修复单层图中的孤立节点。
        // enter_node_ids: BFS 入口（通常为 start point）；node_num: 参与统计的节点数。
        // 返回值：修复前不可从入口到达的节点数量。
        template <typename indexType, typename rangeType>
        static size_t repair_isolate_node(Graph<indexType> &G, rangeType &enter_node_ids, size_t node_num,
                                          size_t batch_size = 1000)
        {
            auto flag = std::vector<int>(G.size(), 0);
            std::vector<size_t> seeds;
            seeds.reserve(enter_node_ids.size());
            for (auto eid : enter_node_ids)
                seeds.push_back(static_cast<size_t>(eid));

            mark_reachable_nodes(G, seeds, flag, batch_size);

            size_t cnt = 0;
            for (size_t i = 0; i < node_num; ++i)
            {
                assert(i < G.size());
                if (flag[i] == 0)
                {
                    ++cnt;
                    try_repair_isolated_node(G, i, flag, enter_node_ids);
                }
            }
            return cnt;
        }

        // 修复 HNSW 分层图中的孤立节点。
        // hidMapping: 层级 id 映射；node_num 通常取 hidMapping->cur_node_num。
        // 仅统计并修复 level=0 的底图节点（向量层），上层辅助节点跳过。
        template <typename indexType, typename rangeType, typename HierMapping>
        static size_t repair_isolate_node(Graph<indexType> &G, rangeType &enter_node_ids, size_t node_num,
                                          size_t batch_size, HierMapping *hidMapping)
        {
            auto flag = std::vector<int>(G.size(), 0);
            std::vector<size_t> seeds;
            seeds.reserve(enter_node_ids.size());
            for (auto eid : enter_node_ids)
                seeds.push_back(static_cast<size_t>(eid));

            mark_reachable_nodes_hier(G, hidMapping, seeds, flag, batch_size);

            size_t cnt = 0;
            for (size_t i = 0; i < node_num; ++i)
            {
                assert(i < G.size());
                if (flag[i] == 0)
                {
                    if (hidMapping->get_level(static_cast<int64_t>(i)) > 0)
                        continue;
                    ++cnt;
                    try_repair_isolated_node(G, i, flag, enter_node_ids);
                }
            }
            return cnt;
        }

        // 单层图便捷入口：以单个 start point 作为 BFS 起点
        template <typename indexType>
        static size_t repair_isolate_node(Graph<indexType> &G, size_t sp, size_t batch_size = 1000)
        {
            std::vector<size_t> enter_node_ids = {sp};
            return repair_isolate_node(G, enter_node_ids, G.size(), batch_size);
        }

        // HNSW 分层图便捷入口：以顶层 entry point 作为 BFS 起点
        template <typename indexType, typename HierMapping>
        static size_t repair_isolate_node(Graph<indexType> &G, size_t sp, size_t batch_size,
                                          HierMapping *hidMapping)
        {
            std::vector<size_t> enter_node_ids = {sp};
            return repair_isolate_node(G, enter_node_ids, hidMapping->cur_node_num, batch_size, hidMapping);
        }
    };

} // namespace ant
