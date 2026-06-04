#pragma once
#include <unordered_set>
#include "nsw.h"

namespace ant
{

    template <typename indexType>
    class FIGM_NSWIndex : public NSWIndex<indexType>
    {
    public:
        // using Point = typename _PR::Point;
        // using PR = _PR;
        using QNode = QueryNode<indexType>;
        using IDist = IdDist<indexType>;

        using SearchFun = std::function<void(indexType, indexType *, uint32_t, BeamSearchMemoryCell<indexType> &)>;
        using UpdateFun = std::function<void(indexType, IDist *, size_t, size_t &)>;
        using FillStartFun = std::function<void(std::vector<std::vector<indexType>> &, size_t, size_t)>;

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
                      float maxCosAngle) : NSWIndex<indexType>(prune, maxBeamSize, max_visited_size, num_workers,
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

            using SearchResult = typename NSWIndex<indexType>::SearchResult;
            auto less = [&](const QNode &a, const QNode &b)
            {
                return id_dist_less(a, b);
            };
            auto unique_less = [&](QNode& a, QNode& b){
                return a.index < b.index;
            };

            auto searchNghAndPrune = [&](indexType index, indexType *starting_points, uint32_t sp_num, BeamSearchMemoryCell<indexType> &bsCell)
            {
                auto [alpha, cos_angle] = this->getAlphaAndCosAngle(index, nodePopular);

                assert(index >= 0 && index < fromG.size());
                QueryParams QP(0, beamSize, 0.0, limit, this->max_degree);
                if (sp_num > beamSize)
                {
                    std::cout << "sp_num=" << sp_num << " beamSize=" << beamSize << " sp_num must less then beamSize!" << std::endl;
                    abort();
                }

                std::vector<QNode> res;
                // 添加旧有邻居
                auto ngh = fromG[index];
                for (size_t i = 0; i < ngh.size(); i++)
                {
                    res.push_back(QNode(ngh[i], Points[ngh[i]].distance(Points[index])));
                }                


                for(int i = 0; i < sp_num; ++i){
                    bsCell.clear();
                    // 搜索邻居
                    bsCell.init_starting_points(Points[index], Points, starting_points, sp_num);

                    size_t bs_distance_comps = 0;

                    auto curQP = QP;

                    bs_distance_comps = beam_search(bsCell, fromG, Points[index], Points, curQP);

                    if (BuildStats != nullptr)
                    {
                        BuildStats->increment_dist(index, bs_distance_comps);
                        BuildStats->increment_visited(index, bsCell.num_visited);
                    }

                    for(auto j = 0; j < bsCell.frontier_size; ++j){
                        res.push_back(bsCell.frontier[j]);
                    }
                }
                std::sort(res.begin(), res.end(), less);
                res.erase(std::unique(res.begin(), res.end(), unique_less),
                                      res.end());
                if(res.size() > bsCell.max_visited_size)
                    res.resize(bsCell.max_visited_size);
                memcpy(bsCell.copy_visited, res.data(), sizeof(QNode) * res.size());
                bsCell.frontier_size = res.size();
                this->prune.sort_candidates(index, fromG, bsCell.copy_visited, bsCell.frontier_size, dir_bias_scale);

                // 剪枝
                size_t start_prune_id = 1;
                auto rp_distance_comps = this->prune.robustPrune(index, bsCell.copy_visited, bsCell.frontier_size, Points, alpha, cos_angle, start_prune_id);
                assert(bsCell.frontier_size <= prune.R);

                if (BuildStats != nullptr)
                    BuildStats->increment_dist(index, rp_distance_comps);
            };

            auto updateDist = [&](indexType index, IDist *candidates, size_t pre_candidate_size, size_t &candidate_size)
            {
                this->updateNeighborsForGraph(Points, index, candidates, pre_candidate_size, candidate_size, BuildStats);
            };
            // ---------------------------------------------------------------------------------

            size_t n = fromG.size();
            size_t m = insert_size;
            size_t inc = 0;
            size_t count = 0;
            float frac = 0.0;
            float progress_inc = .1;
            size_t max_batch_size = std::min(this->getMaxBatchSize(fromG, max_fraction), this->max_batch_size);

            Timer t_beam("beam search time");
            Timer t_bidirect("bidirect time");
            Timer t_prune("prune time");

            t_beam.stop();
            t_bidirect.stop();
            t_prune.stop();

            // cache start points for current batch
            std::vector<std::vector<indexType>> sp_cache = std::vector<std::vector<indexType>>(max_batch_size);
            for(auto i = 0; i < max_batch_size; ++i){
                for(indexType j = 0; j < num_split; ++j){
                    sp_cache[i].push_back(j);
                }
            }

            if (nodePopular == NodePopular::CalPopular){
                this->popular.resize(fromG.size());
                memset(this->popular.data(), 0, fromG.size() * sizeof(float));
            }

            while (count < m)
            {
                size_t floor;
                size_t ceiling;
                if (pow(base, inc) <= max_batch_size)
                {
                    floor = static_cast<size_t>(pow(base, inc)) - 1;
                    ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)) - 1, m);
                    count = ceiling;
                }
                else
                {
                    floor = count;
                    ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
                    count += (ceiling - floor);
                }

                std::vector<std::vector<QNode>> all_new_out_(ceiling - floor);

                std::vector<SearchResult> new_out_(ceiling - floor);

                t_beam.start();
                this->batch_search(new_out_, inserts, floor, ceiling, this->bsTable, sp_cache, searchNghAndPrune);

                // 更新图结构
                {
                    this->batch_update_graph(toG, new_out_);
                    t_beam.stop();

                    // 反转边
                    t_bidirect.start();
                    this->inv_edges(this->merge_edges, new_out_);
                    t_bidirect.stop();

                    // 对反转边进行剪枝
                    t_prune.start();
                    this->batch_update_graph(toG, this->merge_edges, updateDist, this->max_degree);
                    t_prune.stop();
                }

                if (print)
                {
                    auto ind = frac * m;
                    if (floor <= ind && ceiling > ind)
                    {
                        frac += progress_inc;
                        std::cout << "Pass " << (int)(100 * frac) << "% complete"
                                  << std::endl;
                    }
                }
                inc += 1;
            }

            if(nodePopular == NodePopular::CalPopular){
                std::transform(this->popular.begin(), this->popular.end(), this->popular.begin(), [](float a){return log(a+1);});
                auto [min_it, max_it] = std::minmax_element(this->popular.begin(), this->popular.end());
                this->minPopular = (*min_it);
                this->maxPopular = (*max_it);
                std::cout<<this->minPopular<<","<<this->maxPopular<<std::endl;
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