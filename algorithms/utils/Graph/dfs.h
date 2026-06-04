#pragma once
#include <limits>
#include <algorithm>
#include <queue>
#include "graph.h"

namespace ant
{

    struct DFS
    {

        template <typename indexType>
        static void get_popular_nodes(std::vector<indexType> &node_ids, Graph<indexType> &G, std::vector<std::pair<indexType, float>> &tmp_node_popular, float percent)
        {
            tmp_node_popular.resize(G.size());
            memset(tmp_node_popular.data(), 0, G.size() * sizeof(std::pair<indexType, float>));
            for (size_t i = 0; i < tmp_node_popular.size(); ++i)
            {
                tmp_node_popular[i].first = i;
                tmp_node_popular[i].second = G[i].size();
            }
            std::sort(tmp_node_popular.begin(), tmp_node_popular.end(), [](std::pair<indexType, float> a, std::pair<indexType, float> b)
                      { return (a.second < b.second) || (a.second == b.second && a.first < b.first); });

            size_t top_size = tmp_node_popular.size() * percent * 0.5;
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

        template <typename indexType, typename rangeType>
        static size_t repair_isolate_node(Graph<indexType> &G, rangeType &enter_node_ids, size_t node_num, size_t batch_size = 1000)
        {
            auto flag = std::vector<int>(G.size(), 0);
            size_t cnt = 0;
            std::queue<size_t> next_visited;

            for (auto eid : enter_node_ids)
            {
                flag[eid] = 1;
                next_visited.push(eid);
            }

            std::vector<size_t> cur_visited = std::vector<size_t>(batch_size);
            size_t cur_visited_size = 0;

            while (!next_visited.empty())
            {
                cur_visited_size = 0;
                while (cur_visited_size < batch_size && !next_visited.empty())
                {
                    auto cur_node_id = next_visited.front();
                    cur_visited[cur_visited_size++] = cur_node_id;
                    next_visited.pop();
                }

                for (size_t i = 0; i < cur_visited_size; ++i)
                {
                    auto cur_node_id = cur_visited[i];

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

            for (size_t i = 0; i < node_num; ++i)
            {
                assert(i < G.size());
                if (flag[i] == 0)
                {
                    cnt++;
                    auto edges = G[i];
                    for (size_t j = 0; j < edges.size(); ++j)
                    {
                        size_t nid = static_cast<size_t>(edges[j]);
                        auto n_edges = G[nid];
                        if (n_edges.size() < G.max_degree())
                        {
                            n_edges.append_neighbor(i);
                            break;
                        }
                    }
                }
            }
            return cnt;
        }

        template <typename indexType>
        static size_t repair_isolate_node(Graph<indexType> &G, size_t sp, size_t batch_size = 1000)
        {
            std::vector<size_t> enter_node_ids = {sp};
            return repair_isolate_node(G, enter_node_ids, G.size(), batch_size);
        }
    };

}