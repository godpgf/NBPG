#pragma once
#include "nsw.h"

namespace ant
{

    // FIGM 构图：在 fromG 上搜索候选，将剪枝结果写入 toG（双图增量更新）
    template <typename indexType>
    class FIGM_NSWIndex : public NSWIndex<indexType>
    {
    public:
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;
        using SearchResult = typename NSWIndex<indexType>::SearchResult;

        FIGM_NSWIndex(Prune prune,
                      uint32_t maxBeamSize,
                      uint32_t max_visited_size,
                      uint32_t num_workers,
                      uint32_t max_degree,
                      size_t max_batch_size,
                      uint32_t top_degree,
                      float minAlpha,
                      float maxAlpha,
                      float minCosAngle,
                      float maxCosAngle)
            : NSWIndex<indexType>(prune, maxBeamSize, max_visited_size, num_workers,
                                  max_degree, max_batch_size, top_degree, minAlpha, maxAlpha,
                                  minCosAngle, maxCosAngle)
        {
        }

        template <typename PR>
        void pair_batch_insert(indexType *inserts, size_t insert_size, Graph<indexType> &fromG, Graph<indexType> &toG, PR &Points,
                               uint32_t beamSize, size_t limit, QueryStatic *BuildStats, uint32_t num_split,
                               NodePopular nodePopular = NodePopular::FixedMaxPopular, float dir_bias_scale = 0,
                               double base = 2, double max_fraction = .02, bool print = true)
        {
            const auto less = [](const QNode &a, const QNode &b)
            { return id_dist_less(a, b); };
            const auto unique_by_index = [](const QNode &a, const QNode &b)
            { return a.index < b.index; };

            auto searchNghAndPrune = [&](indexType index, indexType *starting_points, uint32_t sp_num, BeamSearchMemoryCell<indexType> &bsCell)
            {
                const auto [alpha, cos_angle] = this->getAlphaAndCosAngle(index, nodePopular);

                assert(index >= 0 && index < fromG.size());
                QueryParams QP(0, beamSize, 0.0, limit, this->max_degree);
                if (sp_num > beamSize)
                {
                    std::cout << "sp_num=" << sp_num << " beamSize=" << beamSize << " sp_num must be less than beamSize!" << std::endl;
                    abort();
                }

                const auto ngh = fromG[index];
                std::vector<QNode> candidates;
                candidates.reserve(ngh.size() + static_cast<size_t>(sp_num) * beamSize);
                for (size_t i = 0; i < ngh.size(); ++i)
                {
                    candidates.emplace_back(ngh[i], Points[ngh[i]].distance(Points[index]));
                }

                for (uint32_t i = 0; i < sp_num; ++i)
                {
                    bsCell.clear();
                    bsCell.init_starting_points(Points[index], Points, starting_points, sp_num);

                    auto curQP = QP;
                    const size_t bs_distance_comps = beam_search(bsCell, fromG, Points[index], Points, curQP);

                    if (BuildStats != nullptr)
                    {
                        BuildStats->increment_dist(index, bs_distance_comps);
                        BuildStats->increment_visited(index, bsCell.num_visited);
                    }

                    for (size_t j = 0; j < bsCell.frontier_size; ++j)
                    {
                        candidates.push_back(bsCell.frontier[j]);
                    }
                }

                std::sort(candidates.begin(), candidates.end(), less);
                candidates.erase(std::unique(candidates.begin(), candidates.end(), unique_by_index), candidates.end());
                if (candidates.size() > bsCell.max_visited_size)
                {
                    candidates.resize(bsCell.max_visited_size);
                }

                bsCell.frontier_size = candidates.size();
                std::memcpy(bsCell.copy_visited, candidates.data(), sizeof(QNode) * candidates.size());
                this->prune.sort_candidates(index, fromG, bsCell.copy_visited, bsCell.frontier_size, dir_bias_scale);

                const size_t start_prune_id = 1;
                const size_t rp_distance_comps = this->prune.robustPrune(
                    index, bsCell.copy_visited, bsCell.frontier_size, Points, alpha, cos_angle, start_prune_id);
                assert(bsCell.frontier_size <= this->prune.R);

                if (BuildStats != nullptr)
                {
                    BuildStats->increment_dist(index, rp_distance_comps);
                }
            };

            auto updateDist = [&](indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size)
            {
                this->updateNeighborsForGraph(Points, index, candidates, pre_candidate_size, candidate_size, BuildStats);
            };

            size_t inc = 0;
            size_t count = 0;
            float frac = 0.0f;
            const size_t effective_batch_size = std::min(this->getMaxBatchSize(fromG, max_fraction), this->max_batch_size);

            Timer t_beam("beam search time");
            Timer t_bidirect("bidirect time");
            Timer t_prune("prune time");
            t_beam.stop();
            t_bidirect.stop();
            t_prune.stop();

            // 每个 batch 槽位固定 num_split 个起始点：0 .. num_split-1
            std::vector<std::vector<indexType>> sp_cache(effective_batch_size);
            for (size_t i = 0; i < effective_batch_size; ++i)
            {
                sp_cache[i].resize(num_split);
                std::iota(sp_cache[i].begin(), sp_cache[i].end(), static_cast<indexType>(0));
            }

            if (nodePopular == NodePopular::CalPopular)
            {
                this->popular.resize(fromG.size());
                std::memset(this->popular.data(), 0, fromG.size() * sizeof(float));
            }

            while (count < insert_size)
            {
                const auto batch = next_batch_range(inc, count, insert_size, effective_batch_size, base);
                const size_t floor = batch.floor;
                const size_t ceiling = batch.ceiling;
                count = batch.next_count;

                std::vector<SearchResult> new_out_(ceiling - floor);

                t_beam.start();
                this->batch_search(new_out_, inserts, floor, ceiling, this->bsTable, sp_cache, searchNghAndPrune);
                if (nodePopular == NodePopular::CalPopular)
                {
                    for (const SearchResult &res : new_out_)
                    {
                        for (size_t i = 0; i < res.vit_size; ++i)
                        {
                            this->popular[res.vit_res[i].index] += 1.0f;
                        }
                    }
                }

                this->batch_update_graph(toG, new_out_);
                t_beam.stop();

                t_bidirect.start();
                this->inv_edges(this->merge_edges, new_out_);
                t_bidirect.stop();

                t_prune.start();
                this->batch_update_graph(toG, this->merge_edges, updateDist, this->max_degree);
                t_prune.stop();

                if (print)
                {
                    report_batch_progress(floor, ceiling, insert_size, frac);
                }
                inc += 1;
            }

            if (nodePopular == NodePopular::CalPopular)
            {
                std::transform(this->popular.begin(), this->popular.end(), this->popular.begin(),
                               [](float v)
                               { return log(v + 1); });
                const auto [min_it, max_it] = std::minmax_element(this->popular.begin(), this->popular.end());
                this->minPopular = *min_it;
                this->maxPopular = *max_it;
                std::cout << this->minPopular << "," << this->maxPopular << std::endl;
            }

            if (print)
            {
                t_beam.total();
                t_bidirect.total();
                t_prune.total();
            }
        }
    };
}
