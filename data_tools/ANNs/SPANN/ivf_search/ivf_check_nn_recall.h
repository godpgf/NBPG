#pragma once
#include "base/check_nn_recall.h"
#include "ivf_beamSearch.h"

namespace ant
{
    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename QCR, typename indexType>
    NN_Result ivf_checkRecall(const Graph<indexType> &G,
                              std::vector<std::vector<uint32_t>> &ivf_index,
                              int32_t bin_size,
                              PR &Base_Points,
                              QyPR &Query_Points,
                              QPR &Q_Base_Points,
                              QCR &Q_Centroids,
                              const std::vector<indexType> &c2v_ids,
                              QyQPR &Q_Query_Points,
                              const GroundTruth<indexType> &GT,
                              const uint32_t k,
                              std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                              const QueryParams &QP,
                              const bool verbose)
    {
        if (GT.size() > 0 && k > GT.dimension())
        {
            std::cout << k << "@" << k << " too large for ground truth data of size "
                      << GT.dimension() << std::endl;
            abort();
        }

        QueryStatic QueryStats(Query_Points.size());
        QueryStats.clear();
        Base_Points.init_cache();

        Timer t;
        auto all_ngh = ivf_qsearchAll<indexType, QyPR, QyQPR, PR, QPR, QCR>(
            Query_Points, Q_Query_Points, G, ivf_index, bin_size,
            Base_Points, Q_Base_Points, Q_Centroids, c2v_ids,
            QueryStats, get_sp, QP);
        const float query_time = t.get_next();

        auto res = calRecall(all_ngh, Base_Points, Query_Points, GT, k, QP, verbose, QueryStats, query_time);
        Base_Points.clear_cache();
        return res;
    }

    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename QCR, typename indexType>
    NN_Result ivf_checkRecall(const Graph<indexType> &G,
                              IVFReader<indexType> &ivfReader,
                              PR &Base_Points,
                              QyPR &Query_Points,
                              QPR &Q_Base_Points,
                              QCR &Q_Centroids,
                              const std::vector<indexType> &c2v_ids,
                              QyQPR &Q_Query_Points,
                              const GroundTruth<indexType> &GT,
                              const uint32_t k,
                              std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                              const QueryParams &QP,
                              const bool verbose)
    {
        if (GT.size() > 0 && k > GT.dimension())
        {
            std::cout << k << "@" << k << " too large for ground truth data of size "
                      << GT.dimension() << std::endl;
            abort();
        }

        QueryStatic QueryStats(Query_Points.size());
        QueryStats.clear();
        Base_Points.init_cache();

        Timer t;
        auto all_ngh = ivf_qsearchAll<indexType, QyPR, QyQPR, PR, QPR, QCR>(
            Query_Points, Q_Query_Points, G, ivfReader,
            Base_Points, Q_Base_Points, Q_Centroids, c2v_ids,
            QueryStats, get_sp, QP);
        const float query_time = t.get_next();

        auto res = calRecall(all_ngh, Base_Points, Query_Points, GT, k, QP, verbose, QueryStats, query_time);
        Base_Points.clear_cache();
        return res;
    }

    template <typename PR, typename QyPR, typename QyQPR, typename QCR, typename indexType>
    NN_Result ivf_checkRecall(const Graph<indexType> &G,
                              IVFPosReader<typename QCR::T, indexType> &ivfPosReader,
                              PR &Base_Points,
                              QyPR &Query_Points,
                              QCR &Q_Centroids,
                              const std::vector<indexType> &c2v_ids,
                              QyQPR &Q_Query_Points,
                              const GroundTruth<indexType> &GT,
                              const uint32_t k,
                              std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                              const QueryParams &QP,
                              const bool verbose)
    {
        if (GT.size() > 0 && k > GT.dimension())
        {
            std::cout << k << "@" << k << " too large for ground truth data of size "
                      << GT.dimension() << std::endl;
            abort();
        }

        QueryStatic QueryStats(Query_Points.size());
        QueryStats.clear();
        Base_Points.init_cache();
        ivfPosReader.init();

        Timer t;
        auto all_ngh = ivf_qsearchAll<QyPR, QyQPR, PR, QCR, indexType>(
            Query_Points, Q_Query_Points, G, ivfPosReader,
            Base_Points, Q_Centroids, c2v_ids, QueryStats, get_sp, QP);
        const float query_time = t.get_next();

        auto res = calRecall(all_ngh, Base_Points, Query_Points, GT, k, QP, verbose, QueryStats, query_time);
        Base_Points.clear_cache();
        ivfPosReader.clear();
        return res;
    }

    template <typename PR, typename QyPR, typename QyQPR, typename QCR, typename indexType>
    void ivf_search_and_parse(Graph_ G_,
                              Graph<indexType> &G,
                              const std::string &ivf_file,
                              PR &Base_Points,
                              QyPR &Query_Points,
                              QCR &Q_Centroids,
                              const std::vector<indexType> &c2v_ids,
                              QyQPR &Q_Query_Points,
                              GroundTruth<indexType> GT, const char *res_file, uint32_t k,
                              std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                              bool verbose = false,
                              uint32_t fixed_beam_width = 0,
                              uint32_t rerank_factor = 100,
                              uint32_t ivf_rerank_factor = 40)
    {
        auto ivfPosReader = IVFPosReader<typename QCR::T, indexType>(ivf_file.c_str());
        auto check = [&](const uint32_t cur_k, const QueryParams QP)
        {
            return ivf_checkRecall(G, ivfPosReader,
                                   Base_Points, Query_Points,
                                   Q_Centroids, c2v_ids, Q_Query_Points,
                                   GT, cur_k, get_sp, QP, verbose);
        };
        _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, ivf_rerank_factor, check);
    }

    template <typename PR, typename QPR, typename QyPR, typename QyQPR, typename indexType>
    void ivf_search_and_parse(Graph_ G_,
                              Graph<indexType> &G,
                              std::vector<std::vector<uint32_t>> &ivf_index,
                              uint32_t bin_size,
                              QPR &Q_Base_Points,
                              PR &Base_Points,
                              QyPR &Query_Points,
                              const std::vector<indexType> &c2v_ids,
                              QyQPR &Q_Query_Points,
                              GroundTruth<indexType> GT, const char *res_file, uint32_t k,
                              std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                              bool verbose = false,
                              uint32_t fixed_beam_width = 0,
                              uint32_t rerank_factor = 100,
                              uint32_t ivf_rerank_factor = 40)
    {
        auto Q_Centroids = RefPointRange<QPR, indexType>(c2v_ids.data(), c2v_ids.size(), &Q_Base_Points);

        auto check = [&](const uint32_t cur_k, const QueryParams QP)
        {
            return ivf_checkRecall(G, ivf_index, bin_size,
                                   Base_Points, Query_Points,
                                   Q_Base_Points, Q_Centroids, c2v_ids, Q_Query_Points,
                                   GT, cur_k, get_sp, QP, verbose);
        };
        _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, ivf_rerank_factor, check);
    }

}
