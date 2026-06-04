#pragma once

#include <math.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <set>
#include "utils/utilities.h"
#include "utils/Graph/query_params.h"

namespace ant
{

    template <typename PointRange, typename indexType>
    struct ClusterPID
    {
        using PR = PointRange;
        using edge = std::pair<indexType, indexType>;
        using pid = IdDist<indexType>;

        static std::pair<indexType, indexType> select_two_random(std::vector<indexType> &active_indices,
                                                                 Random &rnd)
        {
            size_t first_index = rnd.ith_rand(0) % active_indices.size();
            size_t second_index_unshifted = rnd.ith_rand(1) % (active_indices.size() - 1);
            size_t second_index = (second_index_unshifted < first_index) ? second_index_unshifted : (second_index_unshifted + 1);

            return {active_indices[first_index], active_indices[second_index]};
        }

        static void naive_neighbors(PR &Points, std::vector<indexType> &active_indices,
                                    size_t maxK, std::vector<std::vector<pid>> &intermediate_edges, QueryStatic &BuildStats)
        {
            auto less = [&](pid a, pid b)
            { return a.dist < b.dist; };
            size_t n = active_indices.size();

#pragma omp parallel for
            for (size_t i = 0; i < n; ++i)
            {
                std::priority_queue<pid, std::vector<pid>, decltype(less)> Q(less);
                size_t index = active_indices[i];
                BuildStats.increment_dist(index, n - 1);
                // tabulate all-pairs distances between the elements in the leaf
                for (size_t j = 0; j < n; j++)
                {
                    if (j != i)
                    {
                        float dist = Points[index].distance(Points[active_indices[j]]);
                        pid e = pid(active_indices[j], dist);
                        if (Q.size() >= maxK)
                        {
                            float topdist = Q.top().dist;
                            if (dist < topdist)
                            {
                                Q.pop();
                                Q.push(e);
                            }
                        }
                        else
                        {
                            Q.push(e);
                        }
                    }
                }
                size_t q = Q.size();
                std::vector<pid> sorted_edges(q);
                for (indexType j = 0; j < q; j++)
                {
                    sorted_edges[j] = Q.top();
                    Q.pop();
                }
                std::reverse(sorted_edges.begin(), sorted_edges.end());
                
                // merge
                std::vector<pid> new_best;
                new_best.reserve(sorted_edges.size() + intermediate_edges[index].size());
                std::merge(
                    sorted_edges.begin(), sorted_edges.end(),  // 第一个有序范围
                    intermediate_edges[index].begin(), intermediate_edges[index].end(),  // 第二个有序范围
                    std::back_inserter(new_best), // 插入迭代器
                    less
                );
                new_best.erase(std::unique(new_best.begin(), new_best.end(), [](pid a, pid b)
                                               { return a.index == b.index; }), new_best.end());
                if(new_best.size() > maxK){
                    new_best.resize(maxK);
                }

                intermediate_edges[index].resize(new_best.size());
                memcpy(intermediate_edges[index].data(), new_best.data(), new_best.size() * sizeof(pid));
            }
        }

        static void random_clustering(PR &Points, std::vector<indexType> &active_indices,
                               Random &rnd, size_t cluster_size, size_t K, 
                               std::vector<std::vector<pid>> &intermediate_edges, QueryStatic &BuildStats)
        {
            if (active_indices.size() <= cluster_size)
                naive_neighbors(Points, active_indices, K, intermediate_edges, BuildStats);
            else
            {
                auto [f, s] = select_two_random(active_indices, rnd);

                auto left_rnd = rnd.fork(0);
                auto right_rnd = rnd.fork(1);

                BuildStats.increment_dist(f, 1);
                if (Points[f].distance(Points[s]) < 1e-5)
                {
                    std::vector<indexType> closer_first;
                    std::vector<indexType> closer_second;
                    for (indexType i = 0; i < active_indices.size(); i++)
                    {
                        if (i < active_indices.size() / 2)
                            closer_first.push_back(active_indices[i]);
                        else
                            closer_second.push_back(active_indices[i]);
                    }
                    auto left_rnd = rnd.fork(0);
                    auto right_rnd = rnd.fork(1);

                    random_clustering(Points, closer_first, left_rnd, cluster_size,
                                              K, intermediate_edges, BuildStats);
                        
                    random_clustering(Points, closer_first, right_rnd, cluster_size,
                                              K, intermediate_edges, BuildStats);
                

                }
                else
                {
                    BuildStats.increment_dist(f, active_indices.size());
                    BuildStats.increment_dist(s, active_indices.size());
                    // Split points based on which of the two points are closer.
                    std::vector<indexType> closer_first;
                    std::vector<indexType> closer_second;

                    std::vector<bool> dis_cmp(active_indices.size());

                    #pragma omp parallel for
                    for(size_t i = 0; i < active_indices.size(); ++i){
                        indexType ind = active_indices[i];
                        float dist_first = Points[ind].distance(Points[f]);
                        float dist_second = Points[ind].distance(Points[s]);
                        dis_cmp[i] = (dist_first <= dist_second);
                    }

                    for(size_t i = 0; i < active_indices.size(); ++i){
                        if(dis_cmp[i]){
                            closer_first.push_back(active_indices[i]);
                        } else {
                            closer_second.push_back(active_indices[i]);
                        }
                    } 

                    random_clustering(Points, closer_first, left_rnd, cluster_size,
                                              K, intermediate_edges, BuildStats);
                        
                    random_clustering(Points, closer_first, right_rnd, cluster_size,
                                              K, intermediate_edges, BuildStats);
                    
                }
            }
        }

        static void random_clustering_wrapper(PR &Points, size_t cluster_size, size_t K, std::vector<std::vector<pid>> &intermediate_edges, QueryStatic &BuildStats)
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<indexType> uni(0, Points.size());

            Random rnd(uni(rng));

            std::vector<indexType> active_indices = std::vector<indexType>(Points.size());
            std::iota(active_indices.begin(), active_indices.end(), 0);

            random_clustering(Points, active_indices, rnd, cluster_size, K, intermediate_edges, BuildStats);
        }

        static void multiple_clustertrees(PR &Points, size_t cluster_size, size_t num_clusters, size_t K,
                                   std::vector<std::vector<pid>> &intermediate_edges, QueryStatic &BuildStats)
        {
            intermediate_edges.resize(Points.size());
            for (size_t i = 0; i < num_clusters; i++)
            {
                random_clustering_wrapper(Points, cluster_size, K, intermediate_edges, BuildStats);
                std::cout << "Cluster " << i << std::endl;
            }
        }
    };

}