#pragma once
#include <memory>
#include <omp.h>
#include <iostream>
#include <cassert>
#include "utils/pq/pivot.h"

namespace ant
{

    template <typename PR, typename indexType>
    void train_kmeans(std::vector<indexType> &centroid_ids, PR &Points, indexType *indices, indexType indices_size, uint pivots_num, uint max_reps = 8, uint init_scale = 10)
    {
        using dataType = typename PR::T;
        using Point = typename PR::Point;
        indexType dim = Points.dimension() * Point::getChunkSize();
        auto num_workers = omp_get_max_threads();
        std::vector<float> depress_cache = std::vector<float>(dim * num_workers);

        Pivot pivot = Pivot(1, pivots_num, dim, 1, 1);

        auto &pivots = pivot.pivots;
        auto &order_ids = pivot.order_ids;


        // 初始化------------------------------------------------------------------------------------
        size_t init_point_num = pivots_num * init_scale;
        size_t max_cache_size = init_point_num;

        if (init_point_num > indices_size)
        {
            init_point_num = indices_size;
        }

        // 开始初始化pivot_num个质心
        std::vector<indexType> rand_indices = std::vector<indexType>(indices_size);
        memcpy(rand_indices.data(), indices, indices_size * sizeof(indexType));
        std::shuffle(rand_indices.begin(), rand_indices.end(), std::mt19937(0));

        {
            // 初始化第一个轴
            Points[rand_indices[0]].depress(pivots.data());

            // 计算每个向量距离之前轴的最远距离
            std::vector<float> all_dist = std::vector<float>(init_point_num, 0);


            std::vector<float> tmp_dist = std::vector<float>(num_workers);

            // 用距离之前的轴最远的向量初始化后面的轴
            for (size_t i = 1; i < pivots_num; ++i)
            {
                float *pre_pivot = pivots.data() + (i - 1) * dim;

// 计算某个向量距离前轴的最近距离
#pragma omp parallel for
                for (size_t jj = 0; jj < init_point_num; ++jj)
                {
                    auto worker_id = omp_get_thread_num();
                    auto *cur_tmp_dist = tmp_dist.data() + worker_id;
                    auto *cur_depress = depress_cache.data() + worker_id * dim;
                    Points[rand_indices[jj]].depress(cur_depress);

                    Pivot::pq_distance<float>(cur_depress, dim, pre_pivot, order_ids.data(), 1, 1, cur_tmp_dist);

                    float *cur_dist = all_dist.data() + jj;
                    if (i == 1)
                    {
                        cur_dist[0] = cur_tmp_dist[0];
                    }
                    else
                    {
                        cur_dist[0] = std::min(cur_dist[0], cur_tmp_dist[0]);
                    }
                }

                // 找到离之前所有轴最远的向量
                auto max_iter = std::max_element(all_dist.begin(), all_dist.end());
                indexType max_id = std::distance(all_dist.begin(), max_iter);

                // 用最远向量填写当前轴
                {
                    float *cur_pivot = pivots.data() + i * dim;
                    Points[rand_indices[max_id]].depress(cur_pivot);
                }

                // printProgressBar((i + 1) / (double)pivots_num);
            }
        }

        // kmeans-----------------------------------------------------------------------
        size_t max_batch = std::min((size_t)pivots_num * 100, (size_t)indices_size);
        std::shuffle(rand_indices.begin(), rand_indices.end(), std::mt19937(0));
 
        for (uint epoch_id = 0; epoch_id < max_reps; ++epoch_id)
        {
            std::vector<indexType> rperm = std::vector<indexType>(indices_size);
            memcpy(rperm.data(), indices, indices_size * sizeof(indexType));
            std::shuffle(rperm.begin(), rperm.end(), std::mt19937(epoch_id));

            // 统计最近某个轴的向量的数量
            std::vector<size_t> pivots_cnt(pivots_num  * num_workers, 0);
            // 统计最近某个轴的向量累加
            std::vector<double> pivots_sum(pivots_num * dim * num_workers, 0);
            double sum_dis = 0;

            std::vector<float> min_dis = std::vector<float>(num_workers);
            std::vector<float> tmp_dis = std::vector<float>(num_workers);
            std::vector<uint> min_pivot_id = std::vector<uint>(num_workers);

            for (size_t sid = 0; sid < max_batch; sid += max_cache_size)
            {
                size_t cur_batch_size = (sid + max_cache_size) > max_batch ? max_batch - sid : max_cache_size;
                std::vector<float> stat_dis = std::vector<float>(cur_batch_size, 0);

     
#pragma omp parallel for
                for (size_t i = sid; i < sid + cur_batch_size; ++i)
                {
                    auto worker_id = omp_get_thread_num();
                    // 计算最近轴
                    auto worker_offset = worker_id;
                    auto *cur_min_dis = min_dis.data() + worker_offset;
                    auto *cur_tmp_dis = tmp_dis.data() + worker_offset;
                    auto *cur_min_pivot_id = min_pivot_id.data() + worker_offset;

                    // 累加最近轴
                    size_t *cur_pivots_cnt = pivots_cnt.data() + worker_id * pivots_num;
                    double *cur_pivots_sum = pivots_sum.data() + worker_id * pivots_num * dim;

                    auto *cur_depress = depress_cache.data() + worker_id * dim;
                    Points[rand_indices[i]].depress(cur_depress);

                    pivot.encode(cur_depress, dim, cur_min_pivot_id, cur_min_dis, cur_tmp_dis);
                    
                    auto pid = cur_min_pivot_id[0];
                    cur_pivots_cnt[pid]++;
                    for(auto j = 0; j < dim; ++j){
                        auto oj = order_ids[j]; 
                        cur_pivots_sum[pid * dim + oj] += cur_depress[oj]; 
                    }

                    // 统计
                    stat_dis[i - sid] = cur_min_dis[0] / dim;
                }


                for (auto dis : stat_dis)
                    sum_dis += dis;
                // printProgressBar((sid + cur_batch_size) / (double)max_batch);
            }

            for (uint i = 1; i < num_workers; ++i)
            {
                size_t offset = i * pivots_num;
                for (uint j = 0; j < pivots_num; ++j)
                    pivots_cnt[j] += pivots_cnt[offset + j];
                offset = i * pivots_num * dim;
                for (uint j = 0; j < pivots_num * dim; ++j)
                    pivots_sum[j] += pivots_sum[offset + j];
            }

#pragma omp parallel for
            for (size_t i = 0; i < pivots_num; ++i)
            {
                auto cnt = pivots_cnt[i];                      // 当前t_chunk的命中数量
    
                auto cur_pid = i;                        // 当前轴id
                // 当前轴
                auto *cur_pivots = pivots.data() + cur_pid * dim ;
                auto *sum_pivots = pivots_sum.data() + cur_pid * dim;
                for (uint j = 0; j < dim; ++j)
                {
                    auto oj = order_ids[j];
                    cur_pivots[oj] = sum_pivots[oj] / (cnt + 1e-5);
                }
            }
        }

        {
            // 寻找质心对应的
            centroid_ids.resize(pivots_num);
            std::vector<float> tmp_dist(indices_size);
            for (size_t i = 0; i < pivots_num; ++i)
            {
                float *cur_pivot = pivots.data() + i * dim;
                for (size_t sid = 0; sid < indices_size; sid += max_cache_size)
                {
                    size_t cur_batch_size = (sid + max_cache_size) > indices_size ? indices_size - sid : max_cache_size;

                    #pragma omp parallel for
                    for (size_t i = sid; i < sid + cur_batch_size; ++i)
                    {
                        auto *cur_depress = depress_cache.data() + omp_get_thread_num() * dim;
                        Points[rand_indices[i]].depress(cur_depress);
                        Pivot::pq_distance<float>(cur_depress, dim, cur_pivot, order_ids.data(), 1, 1, tmp_dist.data() + i);
                    }
                }
                auto min_iter = std::min_element(tmp_dist.begin(), tmp_dist.end());
    
    // 计算位置索引
                indexType min_position = std::distance(tmp_dist.begin(), min_iter);
                centroid_ids[i] = rand_indices[min_position];
            }
        }

    }
}