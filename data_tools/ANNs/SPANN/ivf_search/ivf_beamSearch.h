#pragma once
#include "utils/Graph/beamSearch.h"
#include "disk_ivf.h"

namespace ant
{
    template <typename PR, typename Query, typename indexType>
    void refresh_ngh_dist(PR &Points, const Query &q, std::vector<IdDist<indexType>> &ngh_dist, std::vector<indexType> &ngh, bool unique)
    {
        auto less = [&](IdDist<indexType> a, IdDist<indexType> b)
        {
            return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
        };

        auto deg = 16 * 1024 / Points.params.num_bytes();
        if (deg <= 0)
            deg = 1;
        ngh.resize(ngh_dist.size());

        std::transform(ngh_dist.begin(), ngh_dist.end(), ngh.data(), [](IdDist<indexType> a)
                       { return a.index; });
        if (unique)
        {
            // 1. 排序
            std::sort(ngh.begin(), ngh.end());
            // 2. 去重
            ngh.erase(std::unique(ngh.begin(), ngh.end()),
                      ngh.end());
            ngh_dist.resize(ngh.size());
        }

        for (size_t sid = 0; sid < ngh.size(); sid++)
        {
            Points.preload(sid);
        }

        for (size_t sid = 0; sid < ngh.size(); sid += deg)
        {
            size_t cur_deg = (sid + deg > ngh.size()) ? ngh.size() - sid : deg;
            for (auto j = 0; j < cur_deg; ++j)
            {
                Points[ngh[sid + j]].prefetch();
            }
            for (auto j = 0; j < cur_deg; ++j)
            {
                auto toid = sid + j;
                ngh_dist[toid] = IdDist<indexType>(ngh[toid], Points[ngh[toid]].distance(q));
            }
        }
        std::sort(ngh_dist.begin(), ngh_dist.end(), less);
    }

    template <typename indexType, typename QyPR, typename QyQPR, typename PR, typename QCR>
    std::vector<std::vector<indexType>>
    ivf_qsearchAll_(QyPR &Query_Points,
                    QyQPR &Q_Query_Points,
                    const Graph<indexType> &G,
                    PR &Base_Points,
                    QCR &Q_Centroids,
                    const std::vector<indexType> &c2v_ids,
                    QueryStatic &QueryStats,
                    std::function<void(BeamSearchMemoryCell<indexType> &, std::vector<IdDist<indexType>> &, std::vector<indexType> &, size_t, size_t)> fill_ngh_dist,
                    std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                    const QueryParams &QP, bool use_rerank, bool contain_centroids = true)
    {
        if (QP.k > QP.beamSize)
        {
            std::cout << "Error: beam search parameter Q = " << QP.beamSize
                      << " same size or smaller than k = " << QP.k << std::endl;
            abort();
        }

        // 申请beamSearch所需要使用的空间----------------------------
        uint num_workers = omp_get_max_threads();
        auto bsTable = BeamSearchMemoryTable<indexType>(Query_Points.size(),
                                                        num_workers,
                                                        QP.beamSize,
                                                        QP.beamSize * 8,
                                                        G.max_degree());

        std::vector<std::vector<indexType>> all_neighbors(Query_Points.size());
#pragma omp parallel for
        for (size_t i = 0; i < Query_Points.size(); ++i)
        {
            if (Q_Centroids.size() == 0)
                continue;
            uint worker_id = omp_get_thread_num();
            // 找到候选邻居集合
            auto [starting_points, sp_num] = get_sp(i);
            BeamSearchMemoryCell<indexType> bsCell = bsTable.getCell(i, worker_id, QP.beamSize, QP.beamSize * 8);
            bsCell.init_starting_points(Q_Query_Points[i], Q_Centroids, starting_points, sp_num);

            // 获得查询到的质心
            auto dist_cmps = beam_search(bsCell, G, Q_Query_Points[i], Q_Centroids, QP);

            QueryStats.increment_visited(i, bsCell.num_visited);
            QueryStats.increment_dist(i, dist_cmps);

            const size_t frontier_size = bsCell.frontier_size;
            const size_t num_check = std::min(static_cast<size_t>(QP.k) * QP.ivf_rerank_factor, frontier_size);

            bsCell.clear();

            std::vector<IdDist<indexType>> ngh_dist;
            ngh_dist.reserve(QP.beamSize * 4);
            std::vector<indexType> ngh(std::min(num_check, frontier_size));
            for (size_t j = 0; j < std::min(num_check, frontier_size); ++j)
            {
                ngh[j] = bsCell.frontier[j].index;
                if (contain_centroids)
                {
                    const auto index = c2v_ids[bsCell.frontier[j].index];
                    const float dist = bsCell.frontier[j].dist;
                    ngh_dist.push_back(IdDist<indexType>(index, dist));
                    bsCell.has_been_seen(index);
                }
            }

            fill_ngh_dist(bsCell, ngh_dist, ngh, num_check, i);

            // 利用真实向量进行重排序
            if (ngh_dist.size() > QP.k * QP.rerank_factor)
            {
                // 截断
                ngh_dist.resize((size_t)QP.k * QP.rerank_factor);
            }

            if (use_rerank)
            {
                refresh_ngh_dist(Base_Points, Query_Points[i], ngh_dist, ngh, false);
            }

            ngh.resize(ngh_dist.size());
            std::transform(ngh_dist.begin(), ngh_dist.end(), ngh.data(), [](IdDist<indexType> pair_data)
                           { return pair_data.index; });
            all_neighbors[i] = ngh;
        }

        return all_neighbors;
    }

    template <typename indexType, typename QyPR, typename QyQPR, typename PR, typename QPR, typename QCR>
    std::vector<std::vector<indexType>>
    ivf_qsearchAll(QyPR &Query_Points,
                   QyQPR &Q_Query_Points,
                   const Graph<indexType> &G,
                   std::vector<std::vector<uint32_t>> &ivf_index,
                   int32_t bin_size,
                   PR &Base_Points,
                   QPR &Q_Base_Points,
                   QCR &Q_Centroids,
                   const std::vector<indexType> &c2v_ids,
                   QueryStatic &QueryStats,
                   std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                   const QueryParams &QP, bool contain_centroid = true)
    {
        auto deg = 16 * 1024 / Q_Base_Points.params.num_bytes();
        auto num_workers = omp_get_max_threads();
        std::vector<indexType> cur_neighbors(deg * num_workers);

        auto less = [&](IdDist<indexType> a, IdDist<indexType> b)
        {
            return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
        };

        auto fill_ngh_dist = [&](BeamSearchMemoryCell<indexType> &bsCell, std::vector<IdDist<indexType>> &ngh_dist, std::vector<indexType> &ngh, size_t num_check, size_t i)
        {
            float dist_threshold = (ngh_dist.size() < num_check || num_check < QP.k) ? std::numeric_limits<float>::max() : ngh_dist[num_check - 1].dist;
            num_check = std::max(num_check, (size_t)QP.k);
            // size_t max_cache_size = std::min(Q_Base_Points.getMaxCache(), QQ_Base_Points.getMaxCache());

            // std::vector<indexType> cur_neighbors;
            // cur_neighbors.reserve(deg);
            auto worker_id = omp_get_thread_num();
            auto *cur_neighbors_ptr = cur_neighbors.data() + worker_id * deg;
            uint32_t cur_neighbors_size = 0;

            auto refresh_neighbors = [&](bool force)
            {
                if (cur_neighbors_size >= deg || force)
                {

                    for (auto j = 0; j < cur_neighbors_size; ++j)
                    {
                        auto index = cur_neighbors_ptr[j];
                        Q_Base_Points[index].prefetch();
                    }
                    for (auto j = 0; j < cur_neighbors_size; ++j)
                    {
                        auto index = cur_neighbors_ptr[j];
                        float dist = Q_Base_Points[index].distance(Q_Query_Points[i]);
                        // tmp[index % 16]++;
                        if (dist < dist_threshold)
                        {
                            ngh_dist.push_back(IdDist<indexType>(index, dist));
                        }
                    }
                    assert(i < Q_Query_Points.size());
                    QueryStats.increment_dist(i, cur_neighbors_size);
                    if (ngh_dist.size() > 0)
                    {
                        std::sort(ngh_dist.begin(), ngh_dist.end(), less);
                        ngh_dist.erase(std::unique(ngh_dist.begin(), ngh_dist.end(), [](IdDist<indexType> a, IdDist<indexType> b)
                                                   { return a.index == b.index; }),
                                       ngh_dist.end());
                    }

                    if (ngh_dist.size() > num_check)
                    {
                        assert(num_check > 0);
                        ngh_dist.resize(num_check);
                        dist_threshold = ngh_dist[num_check - 1].dist;
                    }

                    cur_neighbors_size = 0;
                }
            };

            for (const auto cid : ngh)
            {
                for (auto bin_id = 0; bin_id < bin_size; ++bin_id)
                {
                    auto &ivf_data = ivf_index[cid * (size_t)bin_size + bin_id];
                    for (uint ii = 0; ii < ivf_data.size(); ++ii)
                    {
                        auto index = ivf_data[ii] * bin_size + bin_id;
                        if (bsCell.has_been_seen(index))
                            continue;
                        cur_neighbors_ptr[cur_neighbors_size++] = index;
                        refresh_neighbors(false);
                    }
                }
            }

            refresh_neighbors(true);
        };
        bool use_rerank = Base_Points.params.num_bytes() != Q_Base_Points.params.num_bytes();
        return ivf_qsearchAll_<indexType, QyPR, QyQPR, PR, QCR>(
            Query_Points, Q_Query_Points, G,
            Base_Points, Q_Centroids, c2v_ids, QueryStats,
            fill_ngh_dist, get_sp, QP, use_rerank, contain_centroid);
    }

    template <typename indexType, typename QyPR, typename QyQPR, typename PR, typename QPR, typename QCR>
    std::vector<std::vector<indexType>>
    ivf_qsearchAll(QyPR &Query_Points,
                   QyQPR &Q_Query_Points,
                   const Graph<indexType> &G,
                   IVFReader<indexType> &ivfReader,
                   PR &Base_Points,
                   QPR &Q_Base_Points,
                   QCR &Q_Centroids,
                   const std::vector<indexType> &c2v_ids,
                   QueryStatic &QueryStats,
                   std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                   const QueryParams &QP)
    {
        auto less = [&](IdDist<indexType> a, IdDist<indexType> b)
        {
            return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
        };

        auto fill_ngh_dist = [&](BeamSearchMemoryCell<indexType> &bsCell, std::vector<IdDist<indexType>> &ngh_dist, std::vector<indexType> &ngh, size_t num_check, size_t i)
        {
            float dist_threshold = (ngh_dist.size() < num_check) ? std::numeric_limits<float>::max() : ngh_dist[num_check - 1].dist;

            // size_t max_cache_size = std::min(Q_Base_Points.getMaxCache(), QQ_Base_Points.getMaxCache());
            auto deg = 16 * 1024 / Q_Base_Points.params.num_bytes();
            std::vector<indexType> cur_neighbors;
            cur_neighbors.reserve(deg);

            auto refresh_neighbors = [&](bool force)
            {
                if (cur_neighbors.size() >= deg || force)
                {

                    for (auto index : cur_neighbors)
                    {
                        Q_Base_Points[index].prefetch();
                    }
                    for (auto index : cur_neighbors)
                    {
                        float dist = Q_Base_Points[index].distance(Q_Query_Points[i]);
                        if (dist < dist_threshold)
                        {
                            ngh_dist.push_back(IdDist<indexType>(index, dist));
                        }
                    }
                    std::sort(ngh_dist.begin(), ngh_dist.end(), less);
                    ngh_dist.erase(std::unique(ngh_dist.begin(), ngh_dist.end(), [](IdDist<indexType> a, IdDist<indexType> b)
                                               { return a.index == b.index; }),
                                   ngh_dist.end());
                    if (ngh_dist.size() > num_check)
                    {
                        ngh_dist.resize(num_check);
                        dist_threshold = ngh_dist[num_check - 1].dist;
                    }

                    cur_neighbors.resize(0);
                }
            };

            for (const auto cid : ngh)
            {
                auto [ivf_data, ivf_size] = ivfReader[cid];
                for (uint ii = 0; ii < ivf_size; ++ii)
                {
                    auto index = ivf_data[ii];
                    if (bsCell.has_been_seen(index))
                        continue;
                    cur_neighbors.push_back(index);
                    refresh_neighbors(false);
                }
            }
            refresh_neighbors(true);
        };
        bool use_rerank = Base_Points.params.num_bytes() != Q_Base_Points.params.num_bytes();
        return ivf_qsearchAll_<indexType, QyPR, QyQPR, PR, QCR>(
            Query_Points, Q_Query_Points, G,
            Base_Points, Q_Centroids, c2v_ids, QueryStats,
            fill_ngh_dist, get_sp, QP, use_rerank);
    }

    template <typename QyPR, typename QyQPR, typename PR, typename QCR, typename indexType>
    std::vector<std::vector<indexType>>
    ivf_qsearchAll(QyPR &Query_Points,
                   QyQPR &Q_Query_Points,
                   const Graph<indexType> &G,
                   IVFPosReader<typename QCR::T, indexType> &ivfPosReader,
                   PR &Base_Points,
                   QCR &Q_Centroids,
                   const std::vector<indexType> &c2v_ids,
                   QueryStatic &QueryStats,
                   std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                   const QueryParams &QP)
    {
        using dataType = typename QCR::T;
        auto less = [&](IdDist<indexType> a, IdDist<indexType> b)
        {
            return a.dist < b.dist || (a.dist == b.dist && a.index < b.index);
        };
        auto params = Q_Centroids.params;
        using Point = typename QCR::Point;
        using byte = typename QCR::Point::byte;

        auto fill_ngh_dist = [&](BeamSearchMemoryCell<indexType> &bsCell, std::vector<IdDist<indexType>> &ngh_dist, std::vector<indexType> &ngh, size_t num_check, size_t i)
        {
            float dist_threshold = (ngh_dist.size() < num_check) ? std::numeric_limits<float>::max() : ngh_dist[num_check - 1].dist;

            auto deg = 16 * 1024 / Q_Centroids.params.num_bytes();
            std::vector<std::pair<indexType, dataType *>> cur_neighbors;
            cur_neighbors.reserve(deg);

            auto refresh_neighbors = [&](bool force)
            {
                if (cur_neighbors.size() >= deg || force)
                {

                    for (auto nb : cur_neighbors)
                    {
                        auto p = Point((byte *)nb.second, params);
                        p.prefetch();
                    }
                    for (auto nb : cur_neighbors)
                    {
                        auto p = Point((byte *)nb.second, params);
                        float dist = p.distance(Q_Query_Points[i]);
                        if (dist < dist_threshold)
                        {
                            ngh_dist.push_back(IdDist<indexType>(nb.first, dist));
                        }
                    }
                    std::sort(ngh_dist.begin(), ngh_dist.end(), less);
                    ngh_dist.erase(std::unique(ngh_dist.begin(), ngh_dist.end(), [](IdDist<indexType> a, IdDist<indexType> b)
                                               { return a.index == b.index; }),
                                   ngh_dist.end());
                    if (ngh_dist.size() > num_check)
                    {
                        ngh_dist.resize(num_check);
                        dist_threshold = ngh_dist[num_check - 1].dist;
                    }

                    cur_neighbors.resize(0);
                }
            };

            std::vector<std::pair<indexType *, dataType *>> cache;
            for (auto cid : ngh)
                cache.push_back(ivfPosReader.load_ivf_and_pos(cid));
            size_t dims = ivfPosReader.dims;

            for (size_t j = 0; j < cache.size(); ++j)
            {
                auto [ivf_data, ivf_Pos] = cache[j];
                size_t ivf_size = ((indexType *)ivf_Pos) - ivf_data;
                for (uint ii = 0; ii < ivf_size; ++ii)
                {
                    auto index = ivf_data[ii];
                    if (bsCell.has_been_seen(index))
                        continue;
                    cur_neighbors.push_back(std::make_pair(index, ivf_Pos + ii * dims));
                    refresh_neighbors(false);
                }
            }
            refresh_neighbors(true);
        };
        bool use_rerank = Base_Points.params.num_bytes() != Q_Centroids.params.num_bytes();
        return ivf_qsearchAll_<indexType, QyPR, QyQPR, PR, QCR>(
            Query_Points, Q_Query_Points, G,
            Base_Points, Q_Centroids, c2v_ids, QueryStats,
            fill_ngh_dist, get_sp, QP, use_rerank);
    }

}