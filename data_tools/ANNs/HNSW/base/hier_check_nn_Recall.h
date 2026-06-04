#pragma once
#include "base/check_nn_recall.h"
#include "hier_beamSearch.h"

namespace ant
{

    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename indexType>
    NN_Result hier_checkRecall(const Graph<indexType> &G,
                               HIdMapping &hidMapping,
                               PR &Base_Points,
                               const QyPR &Query_Points,
                               const QPR &Q_Base_Points,
                               const QyQPR &Q_Query_Points,
                               const GroundTruth<indexType> &GT,
                               const uint32_t k,
                               std::function<std::pair<indexType *, uint32_t>(indexType)> get_sp,
                               const QueryParams &QP,
                               const bool verbose,
                               uint32_t hier_beamSize = 1)
    {
        if (GT.size() > 0 && k > GT.dimension())
        {
            std::cout << k << "@" << k << " too large for ground truth data of size "
                      << GT.dimension() << std::endl;
            abort();
        }

        float query_time;
        QueryStatic QueryStats(Query_Points.size());
        QueryStats.clear();

        Base_Points.init_cache();
        Timer t;
        std::vector<std::vector<indexType>> all_ngh = hier_qsearchAll<indexType, QyPR, QyQPR, PR, QPR, Graph<indexType>>(Query_Points, Q_Query_Points, G, hidMapping, Base_Points, Q_Base_Points, QueryStats, get_sp, QP, hier_beamSize);
        query_time = t.get_next();

        auto res = calRecall(all_ngh, Base_Points, Query_Points, GT, k, QP, verbose, QueryStats, query_time);
        Base_Points.clear_cache();
        return res;
    }

    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename indexType>
    void hier_search_and_parse(Graph_ G_,
                               Graph<indexType> &G,
                               HIdMapping &hidMapping,
                               PR &Base_Points,
                               QyPR &Query_Points,
                               QPR &Q_Base_Points,
                               QyQPR &Q_Query_Points,
                               GroundTruth<indexType> GT, const char *res_file, uint32_t k,
                               bool verbose = false,
                               uint32_t fixed_beam_width = 0,
                               uint32_t rerank_factor = 100,
                               uint32_t hier_beamSize = 1)
    {
        indexType sp = hidMapping.get_sp();

        using FType = std::function<std::pair<indexType *, uint32_t>(indexType)>;

        FType get_sp = [&](indexType index) -> std::pair<indexType *, uint32_t>
        {
            return std::make_pair(&sp, 1);
        };

        auto check = [&](const uint32_t k, const QueryParams QP)
        {
            return hier_checkRecall(G, hidMapping,
                                    Base_Points, Query_Points,
                                    Q_Base_Points, Q_Query_Points,
                                    GT, k, get_sp, QP, verbose, hier_beamSize);
        };
        _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, rerank_factor, check);
    }
}