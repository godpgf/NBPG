#pragma once
#include <vector>
#include "utils/utilities.h"
#include "utils/stats/get_time.h"
#include "utils/Graph/query_params.h"

namespace ant
{

    template <typename PointRange, typename indexType>
    struct NNDescent
    {
        using pid = IdDist<indexType>;
        using edge = std::pair<indexType, indexType>;
        using labelled_edge = std::pair<indexType, pid>;

        double delta;
        uint32_t L;
        int max_rounds = 64;

        NNDescent(uint32_t L, double delta, int max_rounds) : L(L), delta(delta), max_rounds(max_rounds)
        {
        }

        template <typename GraphI, typename Prune>
        void undirect_and_prune(GraphI &G, Prune &prune, PointRange &Points, double alpha, double cos_angle, std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            std::vector<std::vector<labelled_edge>> to_group(old_neighbors.size());

            auto less2 = [](pid a, pid b)
            {
                return id_dist_less(a, b);
            };

#pragma omp parallel for
            for (indexType i = 0; i < old_neighbors.size(); ++i)
            {
                size_t s = old_neighbors[i].size();
                std::vector<labelled_edge> e(s);
                for (indexType j = 0; j < s; j++)
                {
                    e[j] = std::make_pair(old_neighbors[i][j].index, pid(i, old_neighbors[i][j].dist));
                }
                to_group[i] = e;
            }

            std::vector<labelled_edge> flat_edges;
            flat(to_group, flat_edges);

            std::vector<std::pair<indexType, std::vector<pid>>> undirected_graph;
            groupby(flat_edges, undirected_graph);

            size_t offset = undirected_graph.size() * delta;
            for (size_t m = 0; m < undirected_graph.size(); m += offset)
            {
                size_t mm = undirected_graph.size() < (m + offset) ? undirected_graph.size() : (m + offset);

#pragma omp parallel for
                for (size_t i = m; i < mm; ++i)
                {
                    indexType index = undirected_graph[i].first;
                    std::vector<pid> &ngh = undirected_graph[i].second;
                    std::vector<pid> new_edges;
                    auto &old_ngh = old_neighbors[index];
                    std::merge(
                        old_ngh.begin(), old_ngh.end(), // 第一个有序范围
                        ngh.begin(), ngh.end(),         // 第二个有序范围
                        std::back_inserter(new_edges),  // 插入迭代器
                        less2);
                    new_edges.erase(std::unique(new_edges.begin(), new_edges.end(), [](auto a, auto b)
                                                { return a.index == b.index; }),
                                    new_edges.end());

                    old_neighbors[index] = new_edges;
                }
            }
            std::cout << std::endl;

// fill new graph ready to beam search
#pragma omp parallel for
            for (size_t index = 0; index < old_neighbors.size(); ++index)
            {
                size_t candidates_size = old_neighbors[index].size();
                if(alpha > 1e-5 || cos_angle > 1e-5){
                    size_t cal_num = prune.robustPrune((indexType)-1, old_neighbors[index].data(), candidates_size, Points, alpha, cos_angle);
                    BuildStats.increment_dist(index, cal_num);
                } 

                if (candidates_size > G.max_degree())
                    candidates_size = G.max_degree();
                G[index].update_neighbors(old_neighbors[index].data(), candidates_size);
            }
        }

        static void reverse_graph(std::vector<std::vector<pid>> &old_neighbors, std::vector<std::pair<indexType, std::vector<indexType>>> &inv_neighbors)
        {

            std::vector<std::vector<edge>> to_group(old_neighbors.size());

#pragma omp parallel for
            for (indexType i = 0; i < old_neighbors.size(); ++i)
            {
                size_t s = old_neighbors[i].size();
                std::vector<edge> e(s);
                for (indexType j = 0; j < s; j++)
                {
                    e[j] = std::make_pair(old_neighbors[i][j].index, i);
                }
                to_group[i] = e;
            }

            std::vector<edge> flat_edges;
            flat(to_group, flat_edges);
            groupby(flat_edges, inv_neighbors);

#pragma omp parallel for
            for (indexType i = 0; i < inv_neighbors.size(); ++i)
            {
                auto &edges = inv_neighbors[i].second;
                std::sort(edges.begin(), edges.end());
                edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
            }
        }

        void nn_descent_chunk(PointRange &Points, std::vector<int> &changed,
                              std::vector<int> &new_changed, std::pair<indexType, std::vector<indexType>> *begin,
                              std::pair<indexType, std::vector<indexType>> *end,
                              std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            size_t stride = end - begin;
            auto less = [](pid a, pid b)
            { return id_dist_less(a, b); };
            std::vector<std::vector<labelled_edge>> grouped_labelled(stride);
            // Timer a("chunk start");
// 从反向边中，找到正向边所没有的邻居，如果这些邻居满足topR最近邻规则约束，就插入
#pragma omp parallel for schedule(auto)
            for (size_t i = 0; i < stride; ++i)
            {
                indexType index = (begin + i)->first;
                std::set<indexType> to_filter;
                to_filter.insert(index);
                for (indexType j = 0; j < old_neighbors[index].size(); j++)
                {
                    to_filter.insert(old_neighbors[index][j].index);
                }
                auto f = [&](indexType a)
                { return (to_filter.find(a) == to_filter.end()); };

                std::vector<indexType> filtered_candidates;
                std::copy_if((begin + i)->second.begin(), (begin + i)->second.end(), std::back_inserter(filtered_candidates), f);

                std::vector<labelled_edge> &edges = grouped_labelled[i];
                edges.reserve(L * 2);
                edges.resize(0);

                size_t cmp_cnt = 0;
                for (indexType l = 0; l < filtered_candidates.size(); l++)
                {
                    indexType j = filtered_candidates[l];
                    float j_max = old_neighbors[j][old_neighbors[j].size() - 1].dist;
                    for (indexType m = l + 1; m < filtered_candidates.size(); m++)
                    {
                        indexType k = filtered_candidates[m];
                        if (changed[j] || changed[k])
                        {
                            float dist = Points[j].distance(Points[k]);
                            cmp_cnt++;
                            float k_max = old_neighbors[k][old_neighbors[k].size() - 1].dist;
                            if (dist < j_max)
                                edges.push_back(std::make_pair(j, pid(k, dist)));
                            if (dist < k_max)
                                edges.push_back(std::make_pair(k, pid(j, dist)));
                        }
                    }
                }
                BuildStats.increment_dist(index, cmp_cnt);
            }
            // a.total();

            // Timer b("chunk groupby");
            // 对所有新插入的边进行groupby
            std::vector<labelled_edge> flat_grouped_labelled;
            flat(grouped_labelled, flat_grouped_labelled);

            std::vector<std::pair<indexType, std::vector<pid>>> candidates;
            groupby(flat_grouped_labelled, candidates);
            // b.total();

            // Timer c("chunk end");
#pragma omp parallel for schedule(auto) // 每个线程每次处理512个迭代
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                auto &ngh = candidates[i].second;
                std::sort(ngh.begin(), ngh.end(), less);
                ngh.erase(std::unique(ngh.begin(), ngh.end(), [](auto a, auto b)
                                      { return a.index == b.index; }),
                          ngh.end());

                std::vector<pid> new_edges;
                indexType index = candidates[i].first;
                auto &old_ngh = old_neighbors[index];
                std::merge(
                    old_ngh.begin(), old_ngh.end(), // 第一个有序范围
                    ngh.begin(), ngh.end(),         // 第二个有序范围
                    std::back_inserter(new_edges),  // 插入迭代器
                    less);
                new_edges.erase(std::unique(new_edges.begin(), new_edges.end(), [](auto a, auto b)
                                            { return a.index == b.index; }),
                                new_edges.end());
                if (new_edges.size() > L)
                {
                    new_edges.resize(L);
                }
                auto last_id = new_edges.size() - 1;
                if (new_edges.size() != old_ngh.size() || new_edges[last_id].index != old_ngh[last_id].index)
                {
                    old_neighbors[index] = new_edges;
                    new_changed[index] = 1;
                }
            }
            // c.total();
        }

        size_t nn_descent(PointRange &Points, std::vector<int> &new_changed, std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            std::vector<int> changed = new_changed;
            memset(new_changed.data(), 0, new_changed.size() * sizeof(int));
            std::vector<std::pair<indexType, std::vector<indexType>>> rev;
            // Timer a("inv");
            reverse_graph(old_neighbors, rev);
            // a.total();

            size_t n = Points.size();
            std::uniform_int_distribution<indexType> dis(0, n - 1);
            int batch_size = 100000;
            auto *begin = rev.data();
            auto *end = rev.data();
            int counter = 0;
            auto *last = rev.data() + rev.size();
            while (end != last)
            {
                counter++;
                begin = end;
                int remaining = last - end;
                end += std::min(remaining, batch_size);
                nn_descent_chunk(Points, changed, new_changed, begin, end, old_neighbors, BuildStats);
            }
            return std::accumulate(new_changed.begin(), new_changed.end(), static_cast<size_t>(0));
        }

        int nn_descent_wrapper(PointRange &Points, std::vector<std::vector<pid>> &old_neighbors, QueryStatic &BuildStats)
        {
            size_t n = Points.size();
            std::vector<int> changed(n, 1);
            int rounds = 0;

            size_t change_num = n;

            while (change_num >= delta * n && rounds < max_rounds)
            {
                change_num = nn_descent(Points, changed, old_neighbors, BuildStats);
                rounds++;
                std::cout << change_num << " elements changed" << std::endl;
                std::cout << "Round " << rounds << " of " << max_rounds << " completed" << std::endl;
            }

            std::cout << "descent converged in " << rounds << " rounds";
            if (rounds < max_rounds)
                std::cout << " (Early termination)";
            std::cout << std::endl;
            return rounds;
        }
    };
}