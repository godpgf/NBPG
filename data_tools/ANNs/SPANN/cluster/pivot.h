#pragma once
#include <omp.h>
#include <limits>
#include <vector>
#include <random>
#include <numeric>

namespace ant
{
    struct Pivot
    {
        Pivot(uint times, uint pivots_num, uint dim, uint chunk, uint sub_chunk, uint seed=0) : times(times), pivots_num(pivots_num), dim(dim), chunk(chunk), sub_chunk(sub_chunk) {
            _init();
            for(uint i = 0; i < times; ++i){
                uint* cur_order_ids = order_ids.data() + i * dim;
                std::iota(cur_order_ids, cur_order_ids + dim, 0);
                // for(uint j = 0; j < dim; ++j)
                    // cur_order_ids[j] = j;
                std::shuffle(cur_order_ids, cur_order_ids + dim, std::mt19937(seed + i));
            }
        }

        Pivot(const char *file)
        {
            std::ifstream reader(file);
            reader.read((char *)&times, sizeof(uint));
            reader.read((char *)&pivots_num, sizeof(uint));
            reader.read((char *)&dim, sizeof(uint));
            reader.read((char *)&chunk, sizeof(uint));
            reader.read((char *)&sub_chunk, sizeof(uint));

            _init();

            reader.read((char *)order_ids.data(), sizeof(float) * times * dim);
            reader.read((char *)pivots.data(), sizeof(float) * times * pivots_num * dim);
            reader.close();
        }

        bool is_combine_dis(){
            // 一个字节有8位，是否将多个距离压缩存储到一个字节？
            return sub_chunk > 1;
        }

        // 计算当前向量与输入轴的距离
        template <typename dataType>
        static void pq_distance(const dataType *values, uint32_t dims, const float *pivots, uint *order_ids, uint times, uint chunk, float *out_dist)
        {
            auto tc = times * chunk;
            auto td = times * dims;
            for (uint j = 0; j < tc; ++j)
            {
                out_dist[j] = 0;
            }
            uint chunk_dim = dims / chunk;
            for (uint j = 0; j < td; j++)
            {
                uint t_chunk_id = j / chunk_dim;
                uint oj = order_ids[j];
                out_dist[t_chunk_id] += (pivots[oj] - values[oj]) * (pivots[oj] - values[oj]);
            }
        }

        void save(const char *file){
            std::ofstream writer(file);
            writer.write((char*)&times, sizeof(uint));
            writer.write((char*)&pivots_num, sizeof(uint));
            writer.write((char*)&dim, sizeof(uint));
            writer.write((char*)&chunk, sizeof(uint));
            writer.write((char*)&sub_chunk, sizeof(uint));
            writer.write((char*)order_ids.data(), sizeof(float) * times * dim);
            writer.write((char*)pivots.data(), sizeof(float) * times * pivots_num * dim);
            writer.close();
        }

        template<typename Point_, typename chunkIndexType>
        void encode(const Point_& point, chunkIndexType* codes, float* min_dis, float* tmp_dist){
            encode(point.data(), point.params.dims, codes, min_dis, tmp_dist);
        }

        template<typename dataType, typename chunkIndexType>
        void encode(const dataType* values, uint32_t dims, chunkIndexType* codes, float* min_dis, float* tmp_dist){
             for(auto scid = 0; scid < stc; ++scid)
                min_dis[scid] = std::numeric_limits<float>::max();
            for(uint pid = 0; pid < pivots_num; ++pid){
                float* cur_pivot = pivots.data() + pid *  td;
                pq_distance(values, dims, cur_pivot, order_ids.data(), times, chunk * sub_chunk, tmp_dist);
                for(uint cid = 0; cid < tc; ++cid){
                    for(uint sid = 0; sid < sub_chunk; ++sid){
                        uint scid = cid * sub_chunk + sid;
                        if(min_dis[scid] > tmp_dist[scid]){
                            min_dis[scid] = tmp_dist[scid];
                            codes[cid * sub_chunk + sid] = pid;
                        }                        
                    }
                }                
            }           
        }

        template<typename chunkIndexType>
        float pq_distance_sum(const float* values, const chunkIndexType* codes){
            uint sub_chunk_dim = dim / (chunk * sub_chunk);
            float out_dist = 0;
            for(auto i = 0; i < tc; ++i){
                auto compress_pid = codes[i];
                for(auto j = 0; j < sub_chunk; ++j){
                    auto pid = (compress_pid & mask);
                    const float* cur_pivots = pivots.data() + pid * td;
                    auto* cur_order_ids = order_ids.data() + (i * sub_chunk + j) * sub_chunk_dim;
                    for(auto jj = 0; jj < sub_chunk_dim; ++jj){
                        uint oi = cur_order_ids[jj];
                        out_dist += (cur_pivots[oi] - values[oi]) * (cur_pivots[oi] - values[oi]);
                    }   
                    compress_pid = (compress_pid >> sub_bit);                   
                }
            }
            return out_dist;
        }

        template<typename Point_, typename chunkIndexType>
        void compress_encode(const Point_& point, chunkIndexType* codes, float* min_dis, float* tmp_dist){
            for(auto scid = 0; scid < stc; ++scid)
                min_dis[scid] = std::numeric_limits<float>::max();
            for(uint pid = 0; pid < pivots_num; ++pid){
                float* cur_pivot = pivots.data() + pid *  td;
                pq_distance(point.data(), point.params.dims, cur_pivot, order_ids.data(), times, chunk * sub_chunk, tmp_dist);
                for(uint cid = 0; cid < tc; ++cid){
                    if(sub_chunk == 1){
                        if(min_dis[cid] > tmp_dist[cid]){
                            codes[cid] = pid;
                            min_dis[cid] = tmp_dist[cid];
                        }
                    } else {
                        for(uint sid = 0; sid < sub_chunk; ++sid){
                            uint scid = cid * sub_chunk + sid;
                            if(min_dis[scid] > tmp_dist[scid]){
                                min_dis[scid] = tmp_dist[scid];
                                uint offset = sid * sub_bit;
                                codes[cid] = (codes[cid] & ~(mask << offset)) + (pid << offset);
                            }                        
                        }
                    }
                }                
            }
        }

        template<typename Point_>
        void preCalculateDis(const Point_& point, float* q2p_dis){
            for(size_t j = 0; j < pivots_num; ++j){
                auto* cur_q2p_dis = q2p_dis + j * stc;
                auto* cur_pivots = pivots.data() + j * td;
                pq_distance(point.data(), point.params.dims, cur_pivots, order_ids.data(), times, chunk * sub_chunk, cur_q2p_dis);
            }        
        }

        // 将sub_chunk中存放的距离组合在一起，以便减少查询次数
        void combineDis(const float* q2p_dis, float* q2cp_dis){
            for(uint tcid = 0; tcid < tc; ++tcid)
                convert2combineDis(q2p_dis, q2cp_dis, tcid);
        }


        float* getPiv2PivDis(){return piv2pivDisCache.data();}

        float* cachePiv2PivDis(){
            uint pp = pivots_num * pivots_num;
            // uint tc = times * chunk;
            // uint td = times * dim;
            uint sub_chunk_dim = dim / (chunk * sub_chunk);
            piv2pivDisCache.reserve(stc * pp);
            float* dis_cache = piv2pivDisCache.data();
            memset(dis_cache, 0, stc * pp * sizeof(float));
    
            // 在一个sub_chunk内计算不同轴之间的距离
            #pragma omp parallel for
            for(size_t stc_id = 0; stc_id < stc; stc_id++){
                float* cur_dis_cache = dis_cache + stc_id * pp;
                uint tid = stc_id / (chunk * sub_chunk);
                uint cid = stc_id % (chunk * sub_chunk);
                uint* cur_order_ids = order_ids.data() + tid * dim + cid * sub_chunk_dim;

                // 计算一个chunk中任意两个轴之间的距离，并保存
                for(uint pi = 0; pi < pivots_num - 1; pi++){
                    float* lp = pivots.data() + pi * td;
                    for(uint pj = pi + 1; pj < pivots_num; pj++){
                        float* rp = pivots.data() + pj * td;
                        float dis = 0;
                        for(int j = 0; j < sub_chunk_dim; ++j){
                            int cj = cur_order_ids[j];
                            dis += (lp[cj] - rp[cj]) * (lp[cj] - rp[cj]);
                        }
                        cur_dis_cache[pi * pivots_num + pj] = dis;
                        cur_dis_cache[pj * pivots_num + pi] = dis;
                    }
                }

            }
            return dis_cache;    
        }

        uint get_sub_bit(){
            return sub_bit;
        }

        uint get_combine_pivots_num(){
            return combine_pivots_num;
        }

        uint dim, times, pivots_num, chunk, sub_chunk, combine_pivots_num;
        std::vector<float> pivots;
        std::vector<uint> order_ids;
        // 缓存某个chunk内任意两个轴心的距离
        std::vector<float> piv2pivDisCache;

    protected:
        void convert2combineDis(const float* q2p_dis, float* q2cp_dis, uint tcid, uint sid=0, uint cur_combine_pivots=0, float sumDis=0){
            if(sid == sub_chunk){
                (q2cp_dis + cur_combine_pivots * tc)[tcid] = sumDis;
            } else {
                for(uint pid = 0; pid < pivots_num; ++pid){
                    uint offset = sid * sub_bit;
                    cur_combine_pivots = (cur_combine_pivots & ~(mask << offset)) + (pid << offset);
                    float curDis = (q2p_dis + pid * stc)[tcid * sub_chunk + sid];
                    convert2combineDis(q2p_dis, q2cp_dis, tcid, sid+1, cur_combine_pivots, sumDis + curDis);
                }
            }
        }

        void _init(){
            pivots = std::vector<float>(times * pivots_num * dim);
            order_ids = std::vector<uint>(times * dim);

            // 设置各种中间数据
            td = times * dim;
            tc = times * chunk;
            stc = times * chunk * sub_chunk;
            sub_dim = dim / (chunk * sub_chunk);
            
            
            // 得到每个sub_chunk需要的位数
            uint tmp_p = pivots_num - 1;
            sub_bit = 0;
            while (tmp_p > 0)
            {
                tmp_p = (tmp_p >> 1);
                ++sub_bit;
            }

            mask = (1 << sub_bit) - 1;

            combine_pivots_num = 0;
            for(uint i = 0; i < sub_chunk; ++i){
                combine_pivots_num = (combine_pivots_num << sub_bit) + (pivots_num - 1);
            }
            combine_pivots_num++;
            // std::cout<<"combine_pivots_num:"<<combine_pivots_num<<std::endl;
        }
        uint td, tc, stc, sub_dim, mask, sub_bit;
    };



}