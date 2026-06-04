#pragma once
#include <math.h>
#include <algorithm>
#include <random>
#include <set>
#include <cstdlib> // 包含 rand() 和 srand()
#include <ctime>   // 包含 time()
#include <omp.h>
#include "query_params.h"

namespace ant
{

    struct Prune
    {
        uint32_t PR, R;

        Prune(uint32_t R, uint32_t num_workers = 1, uint32_t PR=0) : PR(PR), R(R), num_workers(num_workers)
        {
            if (PR > 0)
            {
                tmp_score.resize(num_workers * PR);
            }
        }

        bool usePruneEnhance() { return PR > 0; }

        template <typename id_dist>
        void sort_candidates(id_dist *candidates, size_t& candidates_size){

            auto unique_less = [&](id_dist& a, id_dist& b){
                return a.index < b.index;
            };
            // remove any duplicates
            std::sort(candidates, candidates + candidates_size, unique_less);
            auto new_end = std::unique(candidates, candidates + candidates_size,
                                        [&](auto x, auto y)
                                        { return x.index == y.index; });
            candidates_size = new_end - candidates;

            // Sort the candidate set according to distance from p
            auto less = [&](id_dist& a, id_dist& b)
            {
                return id_dist_less(a, b);
            };
            std::sort(candidates, candidates + candidates_size, less);
        }

        template <typename indexType, typename id_dist, typename graphType>
        void sort_candidates(indexType p, graphType& G, id_dist *candidates, size_t& candidates_size, float dir_bias_scale=0){

            auto unique_less = [&](id_dist& a, id_dist& b){
                return a.index < b.index;
            };
            // remove any duplicates
            std::sort(candidates, candidates + candidates_size, unique_less);
            auto new_end = std::unique(candidates, candidates + candidates_size,
                                        [&](auto x, auto y)
                                        { return x.index == y.index; });
            candidates_size = new_end - candidates;

            // Sort the candidate set according to distance from p
            auto less = [&](id_dist a, id_dist b)
            {
                if(dir_bias_scale > 0){
                    a.dist *= (dir_bias_scale * log(G[a.index].size() + 1) + 1);
                    b.dist *= (dir_bias_scale * log(G[b.index].size() + 1) + 1);
                }
                if(a.index == p)
                    a.dist = std::numeric_limits<float>::max();
                if(b.index == p)
                    b.dist = std::numeric_limits<float>::max();
                return id_dist_less(a, b);
            };
            std::sort(candidates, candidates + candidates_size, less);
            while(candidates_size > 0 && candidates[candidates_size-1].index == p)
                candidates_size--;
        }

        /*template <typename indexType, typename id_dist, typename graphType>
        void depth_sort_candidates(indexType p, graphType& G, id_dist *candidates, size_t& candidates_size, float dir_bias_scale=0){
            if(candidates_size == 0)
                return;

            // 过滤
            size_t nearest_id = -1;
            size_t easy_id = -1;
            
            size_t new_candidates_size = 0;
            for(size_t i = 0; i < candidates_size; ++i){
                // 先过滤掉非叶节点
                if(!candidates[i].isLeaf || candidates[i].index == p)
                    continue;
                candidates[new_candidates_size] = candidates[i];
                // 将深度加一，以便后面进行log运算
                candidates[new_candidates_size].depth++;

                // 先找到距离查询向量最近的节点，以及最容易搜索到的节点
                if(nearest_id == -1 || candidates[new_candidates_size].dist < candidates[nearest_id].dist){
                    nearest_id = new_candidates_size;
                }
                if(easy_id == -1 || candidates[new_candidates_size].depth < candidates[nearest_id].depth){
                    easy_id = new_candidates_size;
                }
                new_candidates_size++;
            }
            candidates_size = new_candidates_size;


            // 将最近的和最容易搜索到的节点深度设置为1，这样排序可以排在最前面
            if(nearest_id != -1)
                candidates[nearest_id].depth = 0;
            if(easy_id != -1)
                candidates[easy_id].depth = 0;

            // 去重
            auto unique_less = [&](id_dist& a, id_dist& b){
                return a.index < b.index;
            };
            // remove any duplicates
            std::sort(candidates, candidates + candidates_size, unique_less);
            auto new_end = std::unique(candidates, candidates + candidates_size,
                                        [&](auto x, auto y)
                                        { return x.index == y.index; });
            candidates_size = new_end - candidates;    
            
            // Sort the candidate set according to distance from p
            auto less = [&](id_dist a, id_dist b)
            {
                // a.dist *= log(a.depth+1.0f);
                // b.dist *= log(b.depth+1.0f);
                // a.dist *= 1.0f / (log(a.depth + 1.0f) + 1);
                // b.dist *= 1.0f / (log(b.depth + 1.0f) + 1);
                // if(dir_bias_scale > 0){
                //     a.dist *= (dir_bias_scale * log(G[a.index].size() + 1) + 1);
                //     b.dist *= (dir_bias_scale * log(G[b.index].size() + 1) + 1);
                // }
                // return log(a.depth+1.0f) * a.dist < log(b.depth+1.0f) * b.dist;
                // return a.dist < b.dist;

                if(dir_bias_scale > 0){
                    a.dist *= (dir_bias_scale * log(G[a.index].size() + 1) + 1);
                    b.dist *= (dir_bias_scale * log(G[b.index].size() + 1) + 1);
                }
                if(a.index == p)
                    a.dist = std::numeric_limits<float>::max();
                if(b.index == p)
                    b.dist = std::numeric_limits<float>::max();
                return id_dist_less(a, b);

            };
            std::sort(candidates, candidates + candidates_size, less);
            
        }*/

        template <typename indexType, typename id_dist, typename PointRange>
        size_t robustPrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points, double alpha, double cos_angle, size_t start_prune_id=1){
            if(usePruneEnhance()){
                return maxAnglePrune(p, candidates, candidates_size, Points, alpha, cos_angle, start_prune_id);
            } else {
                return limitedSimilarityPrune(p, candidates, candidates_size, Points, alpha, cos_angle, start_prune_id);
            }
        }

        template <typename indexType, typename id_dist, typename PointRange>
        size_t maxAnglePrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points, double alpha, double cos_angle, size_t start_prune_id=1){
            // 设C为以选元素集合，NC为未选元素集合，每次从NC中找到与C角度最大的元素，插入C，然后对NC进行基本的剪枝
            assert(PR >= candidates_size);
            auto worker_id = omp_get_thread_num() % num_workers;
            float* score = tmp_score.data() + worker_id * PR;
            size_t distance_comps = 0;

            if(alpha <= 1e-5 && cos_angle <= 1e-5){
                std::cout<<"基于距离的剪枝参数alpha和基于角度的剪枝参数cos_angle，必须设置其中的一个，现在两个都没有设置。"<<std::endl;
                abort();
            }

            auto *new_nbhs = candidates;
            size_t new_nbhs_size = start_prune_id > 0 ? start_prune_id - 1 : 0;
            size_t candidate_idx = 0;     
            
            while (new_nbhs_size < R && new_nbhs_size < candidates_size)
            {
                // 代码运行到此处时要求：score[new_nbhs_size] >= max(score[new_nbhs_size+1:candidates_size])
                
                // （1）选择最大角度的节点，插入到C
                auto& cur_node = candidates[new_nbhs_size];
                indexType p_star = cur_node.index;
                if (p_star == p || p_star == -1)
                {
                    std::cout<<"剪枝时遇到非法节点！"<<std::endl;
                    abort();
                }
                new_nbhs_size++;

                // (2)对NC进行剪枝，没有被减的元素都移动到前面
                size_t new_candidates_size = new_nbhs_size;
                for (size_t i = new_nbhs_size; i < candidates_size; i++){
                    size_t p_prime = candidates[i].index;
                    assert(p_prime != -1);
                    distance_comps++;
                    float dist_starprime = Points[p_star].distance(Points[p_prime]);
                    float dist_pprime = candidates[i].dist;
                    // 计算与当前元素的角度
                    float a2 = dist_starprime;
                    float b2 = dist_pprime;
                    float c2 = cur_node.dist;
                    float b = std::sqrt(b2);
                    float c = std::sqrt(c2);
                    float cur_cos = (b2 + c2 - a2) / (2 * b * c);
                    // 余弦越大，夹角越小，当前元素与C中所有元素中的最小夹角，就是当前元素与C的夹角，所以这里需要记录最小夹角，即最大余弦
                    if(cur_cos > score[i]){
                        score[i] = cur_cos;
                    }

                    if (alpha > 1e-5 && alpha * dist_starprime > dist_pprime)
                    {
                        // 当前元素没有被距离剪枝，移动到前面去
                        if(i > new_candidates_size){
                            std::swap(candidates[new_candidates_size], candidates[i]);
                            std::swap(score[new_candidates_size], score[i]);     
                        }
                    }else if(cos_angle > 1e-5 && cur_cos < cos_angle){
                        // 夹角足够大，当前元素没有被角度剪枝，移动到前面去
                        if(i > new_candidates_size){
                            std::swap(candidates[new_candidates_size], candidates[i]);
                            std::swap(score[new_candidates_size], score[i]);
                        }
                        
                    } 

                    // (3)将与C夹角最大的（余弦最小的）元素移动到最前面
                    if(score[new_candidates_size] < score[new_nbhs_size]){
                        std::swap(candidates[new_candidates_size], candidates[new_nbhs_size]);
                        std::swap(score[new_candidates_size], score[new_nbhs_size]);                        
                    }

                    ++new_candidates_size;
                }
                candidates_size = new_candidates_size;
            }
            candidates_size = new_nbhs_size;
            return distance_comps;
        }

        // 剪枝，将剪枝后的结果存入candidates，candidates_size。注意，此函数会覆盖candidates中原来的数据，请确认candidates在剪枝后就无用
        // score用于我们设计的剪枝增强算法，长度是PR
        // start_prune_id表示从哪个id开始进行剪枝
        // 返回距离计算次数
        template <typename indexType, typename id_dist, typename PointRange>
        size_t limitedSimilarityPrune(indexType p, id_dist *candidates, size_t &candidates_size, PointRange &Points, double alpha, double cos_angle, size_t start_prune_id=1)
        {

            /*float *score = nullptr;
            if (PR > 0)
            {
                auto worker_id = omp_get_thread_num() % num_workers;
                score = tmp_score.data() + worker_id * PR;
            }*/

            size_t distance_comps = 0;

            if (alpha <= 1e-5 && cos_angle <= 1e-5)
            {
                if (candidates_size > R)
                    candidates_size = R;
                return 0;
            }

            auto *new_nbhs = candidates;
            size_t new_nbhs_size = 0;
            size_t candidate_idx = 0;

            while (new_nbhs_size < R && candidate_idx < candidates_size)
            {
                // Don't need to do modifications.
                indexType p_star = candidates[candidate_idx].index;
                candidate_idx++;
                if (p_star == p || p_star == -1)
                {
                    continue;
                }

                new_nbhs[new_nbhs_size++] = candidates[candidate_idx - 1];

                for (size_t i = candidate_idx; i < candidates_size; i++)
                {
                    if(i < start_prune_id)
                        continue;
                    int p_prime = candidates[i].index;
                    if (p_prime != -1)
                    {
                        distance_comps++;
                        float dist_starprime = Points[p_star].distance(Points[p_prime]);
                        float dist_pprime = candidates[i].dist;
                        if (alpha > 1e-5)
                        {
                            if (alpha * dist_starprime <= dist_pprime)
                            {
                                candidates[i].index = -1;
                            }
                        }
                        else if (cos_angle > 1e-5)
                        {
                            float a2 = dist_starprime;
                            float b2 = dist_pprime;
                            float c2 = candidates[candidate_idx - 1].dist;
                            float b = std::sqrt(b2);
                            float c = std::sqrt(c2);
                            float cur_cos = (b2 + c2 - a2) / (2 * b * c);
                            if (cur_cos > cos_angle)
                            {
                                // 夹角越小cos越大，需要把夹角小的（cos大的）删掉
                                candidates[i].index = -1;
                            }
                        }
                    }
                }
            }
            candidates_size = new_nbhs_size;

            /*if (usePruneEnhance() && candidates_size > R)
            {
                // 从PR个元素中选出最好的R个
                new_nbhs_size = 0;

                if (score == nullptr)
                {
                    std::cout << "prune ERROR:score is empty!" << std::endl;
                    abort();
                }
                memset(score, 0, PR * sizeof(float));

                int min_score_id = 0;
                while (new_nbhs_size < R && min_score_id >= 0)
                {
                    if (min_score_id != new_nbhs_size)
                    {
                        // 选出最好的那个元素
                        std::swap(candidates[new_nbhs_size], candidates[min_score_id]);
                        std::swap(score[new_nbhs_size], score[min_score_id]);
                    }
                    auto p_star = new_nbhs[new_nbhs_size].index;
                    float dist_star = new_nbhs[new_nbhs_size].dist;
                    new_nbhs_size++;

                    min_score_id = -1;
                    for (int i = new_nbhs_size; i < candidates_size; ++i)
                    {
                        // 刷新所有其他元素的打分，同时考虑和当前选择的元素的角度和距离来评估打分
                        // 找到打分最小的元素（最好的元素）
                        size_t p_prime = candidates[i].index;
                        distance_comps++;
                        float dist_starprime = Points[p_star].distance(Points[p_prime]);
                        float dist_pprime = candidates[i].dist;
                        float a2 = dist_starprime;
                        float b2 = dist_pprime;
                        float c2 = dist_star;
                        float b = std::sqrt(b2);
                        float c = std::sqrt(c2);
                        float cur_cos = (b2 + c2 - a2) / (2 * b * c);
                        // score[i] = std::max<float>(score[i], (1.0001f + cur_cos) * b);
                        score[i] = std::max<float>(score[i], cur_cos);
                        if (min_score_id < 0 || score[i] < score[min_score_id])
                        {
                            min_score_id = i;
                        }
                    }
                }
                candidates_size = new_nbhs_size;
            }*/

            return distance_comps;
        }

    protected:
        std::vector<float> tmp_score;
        uint32_t num_workers;
    };

}