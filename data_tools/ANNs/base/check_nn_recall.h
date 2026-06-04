#pragma once
#include "utils/Graph/graph.h"
#include "ground_truth.h"
#include "utils/stats/get_time.h"
#include "utils/Graph/beamSearch.h"
#include "parse_results.h"
#include "csv_file.h"
#include "utils/ivf/ivf.h"

namespace ant
{
    void write_to_csv(std::string csv_filename, std::vector<float> buckets,
                      std::vector<NN_Result> results, Graph_ G)
    {
        csvfile csv(csv_filename);
        csv << "GRAPH"
            << "Parameters"
            << "Size"
            << "Build time"
            << "Avg degree"
            << "Max degree"
            << "Isolated"
            << "Popular Avg degree"
            << endrow;
        csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg << G.isolated_cnt << G.popular_avg_deg
            << endrow;
        csv << endrow;
        csv << "Num queries"
            << "Target recall"
            << "Actual recall"
            << "QPS"
            << "Average Cmps"
            << "Tail Cmps"
            << "Average Visited"
            << "Tail Visited"
            << "k"
            << "Q"
            << "cut" << endrow;
        for (int i = 0; i < results.size(); i++)
        {
            NN_Result N = results[i];
            csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
                << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
                << N.cut << endrow;
        }
        csv << endrow;
        csv << endrow;
    }

    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename indexType>
    NN_Result checkRecall(const Graph<indexType> &G,
                          PR &Base_Points,
                          const QyPR &Query_Points,
                          const QPR &Q_Base_Points,
                          const QyQPR &Q_Query_Points,
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

        float query_time;
        QueryStatic QueryStats(Query_Points.size());
        QueryStats.clear();

        Base_Points.init_cache();
        Timer t;
        std::vector<std::vector<indexType>> all_ngh = qsearchAll<indexType, QyPR, QyQPR, PR, QPR, Graph<indexType>>(Query_Points, Q_Query_Points, G, Base_Points, Q_Base_Points, QueryStats, get_sp, QP);
        query_time = t.get_next();

        auto res = calRecall(all_ngh, Base_Points, Query_Points, GT, k, QP, verbose, QueryStats, query_time);
        Base_Points.clear_cache();
        return res;
    }

    template <typename PR, typename QyPR, typename indexType>
    NN_Result calRecall(std::vector<std::vector<indexType>> &all_ngh,
                        PR &Base_Points,
                        const QyPR &Query_Points,
                        const GroundTruth<indexType> &GT,
                        const uint32_t k,
                        const QueryParams &QP,
                        const bool verbose, QueryStatic &QueryStats, float query_time)
    {
        using Point = typename QyPR::Point;
        float recall = 0.0;
        // TODO deprecate this after further testing
        bool dists_present = true;
        if (GT.size() > 0 && !dists_present)
        {
            size_t n = Query_Points.size();
            int numCorrect = 0;
            for (indexType i = 0; i < n; i++)
            {
                std::set<indexType> reported_nbhs;
                if (all_ngh[i].size() != k)
                {
                    std::cout << "bad number of neighbors reported: " << all_ngh[i].size() << std::endl;
                    abort();
                }
                for (indexType l = 0; l < k; l++)
                    reported_nbhs.insert((all_ngh[i])[l]);
                if (reported_nbhs.size() != k)
                {
                    std::cout << "duplicate entries in reported neighbors" << std::endl;
                    abort();
                }
                for (indexType l = 0; l < k; l++)
                {
                    if (reported_nbhs.find((GT.coordinates(i, l))) !=
                        reported_nbhs.end())
                    {
                        numCorrect += 1;
                    }
                }
            }
            recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
        }
        else if (GT.size() > 0 && dists_present)
        {
            size_t n = Query_Points.size();

            int numCorrect = 0;
            for (indexType i = 0; i < n; i++)
            {
                std::vector<int> results_with_ties;
                for (indexType l = 0; l < k; l++)
                    results_with_ties.push_back(GT.coordinates(i, l));
                Point qp = Query_Points[i];
                float last_dist = Base_Points[GT.coordinates(i, k - 1)].distance(qp);
                // float last_dist = GT.distances(i, k-1);
                for (indexType l = k; l < GT.dimension(); l++)
                {
                    // if (GT.distances(i,l) == last_dist) {
                    if (Base_Points[GT.coordinates(i, l)].distance(qp) == last_dist)
                    {
                        results_with_ties.push_back(GT.coordinates(i, l));
                    }
                    else
                    {
                        break;
                    }
                }
                std::set<indexType> reported_nbhs;
                for (indexType l = 0; l < k; l++)
                {
                    if (l < all_ngh[i].size())
                        reported_nbhs.insert((all_ngh[i])[l]);
                }

                for (indexType l = 0; l < results_with_ties.size(); l++)
                {
                    if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end())
                    {
                        numCorrect += 1;
                    }
                }
            }
            recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
        }
        float QPS = Query_Points.size() / query_time;
        auto ds = QueryStats.dist_stats();
        auto vs = QueryStats.visited_stats();
        if (verbose)
            std::cout << "search: Q=" << QP.beamSize << ", k=" << QP.k
                      << ", limit=" << QP.limit
                      //<< ", dlimit=" << QP.degree_limit
                      << ", recall=" << recall
                      << ", visited=" << vs.first
                      << ", comparisons=" << ds.first
                      << ", QPS=" << QPS
                      << ", ctime=" << 1 / (QPS * ds.first) * 1e9 << std::endl;

        std::vector<uint32_t> stats = {ds.first, ds.second, vs.first, vs.second};
        NN_Result N(recall, stats, QPS, k, QP.beamSize, QP.cut, Query_Points.size(), QP.limit, QP.degree_limit, k);
        return N;
    }

    template <typename indexType>
    void _search_and_parse(Graph_ G_,
                           Graph<indexType> &G,
                           const char *res_file, uint32_t k,
                           uint32_t fixed_beam_width,
                           uint32_t rerank_factor,
                           uint32_t ivf_rerank_factor,
                           std::function<NN_Result(const long, const QueryParams)> check)
    {
        std::vector<NN_Result> results;
        std::vector<uint32_t> beams;
        std::vector<uint32_t> allr;
        std::vector<double> cuts;

        QueryParams QP;
        QP.limit = G.size();
        QP.rerank_factor = rerank_factor;
        QP.ivf_rerank_factor = ivf_rerank_factor;
        QP.degree_limit = G.max_degree();
  
        beams = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32,
                 34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160,
                 180, 200, 225, 250, 275, 300, 375, 500, 750, 1000};
        if (k == 0)
            allr = {10};
        else
            allr = {k};
        cuts = {1.35};

        if (fixed_beam_width != 0)
        {
            QP.k = allr[0];
            QP.cut = cuts[0];
            QP.beamSize = fixed_beam_width;
            for (int i = 0; i < 5; i++)
                check(QP.k, QP);
        }
        else
        {
            for (long r : allr)
            {
                results.clear();
                QP.k = r;
                for (float cut : cuts)
                {
                    QP.cut = cut;
                    for (float Q : beams)
                    {
                        QP.beamSize = Q;
                        if (Q >= r)
                        {
                            results.push_back(check(r, QP));
                        }
                    }
                }

                // check "limited accuracy"
                // {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35}; //
                std::vector<long> limits = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 35};

                QP = QueryParams(r, r, 1.35, (long)G.size(), (long)G.max_degree());
                for (long l : limits)
                {
                    QP.limit = l;
                    QP.beamSize = std::max<long>(l, r);
                    // for(long dl : degree_limits){
                    QP.degree_limit = std::min<int>(G.max_degree(), 5 * l);
                    results.push_back(check(r, QP));
                }
                // check "best accuracy"
                QP = QueryParams((long)100, (long)1000, (double)10.0, (long)G.size(), (long)G.max_degree());
                results.push_back(check(r, QP));

                std::vector<float> buckets = {.1, .2, .3, .4, .5, .6, .7, .75, .8, .85,
                                              .9, .93, .95, .97, .98, .99, .995, .999, .9995,
                                              .9999, .99995, .99999};
                auto [res, ret_buckets] = parse_result(results, buckets);
                std::cout << std::endl;
                if (res_file != NULL)
                    write_to_csv(std::string(res_file), ret_buckets, res, G_);
            }
        }
    }

    template <typename PR, typename QyPR, typename QPR, typename QyQPR, typename indexType>
    void search_and_parse(Graph_ G_,
                          Graph<indexType> &G,
                          PR &Base_Points,
                          QyPR &Query_Points,
                          QPR &Q_Base_Points,
                          QyQPR &Q_Query_Points,
                          GroundTruth<indexType> GT, const char *res_file, uint32_t k,
                          bool verbose = false,
                          uint32_t fixed_beam_width = 0,
                          uint32_t rerank_factor = 100)
    {
        indexType sp = 0;
        using FType = std::function<std::pair<indexType *, uint32_t>(indexType)>;

        // size_t qn = 0, qd = 0;
        // std::vector<std::vector<indexType>> sp_cache;

        FType get_sp = [&](indexType index) -> std::pair<indexType *, uint32_t>
        {
            // if (sp_cache.size() == 0)
            // {
                return std::make_pair(&sp, 1);
            // }

            // indexType *ptr = sp_cache[index].data();
            // return std::make_pair(ptr, (uint32_t)sp_cache[index].size());
        };

        auto check = [&](const uint32_t k, const QueryParams QP)
        {
            /*if (sp_file != "")
            {
                std::vector<indexType> start_points;
                std::vector<std::vector<indexType>> ivf;

                std::ifstream reader;
                indexType n, d;
                reader.open(sp_file.c_str(), std::ios::in | std::ios::binary);
                if (!reader.is_open())
                {
                    std::cout << "Data file " << sp_file << " not found" << std::endl;
                    std::abort();
                }
                reader.read((char *)&n, sizeof(size_t));
                reader.read((char *)&d, sizeof(size_t));
                start_points = std::vector<indexType>(n * d);
                sp_cache.resize(n);
                for(int i = 0; i < sp_cache.size(); ++i){
                    sp_cache[i].resize(0);
                }
                reader.read((char *)start_points.data(), n * d * sizeof(indexType));
                reader.close();
                loadVectorBinary(ivf, ivf_file);
                // size_t max_len = QP.beamSize * 3 / (2 * d);
                size_t max_len = 1;

                for (uint32_t j = 0; j < n; ++j)
                {
                    auto *cur_sp = start_points.data() + j * d;
                    for (int jj = 0; jj < d; ++jj)
                    {
                        auto cid = cur_sp[jj];
                        if (cid == -1)
                            break;
                        auto len = std::min(ivf[cid].size(), max_len);
                        sp_cache[j].insert(sp_cache[j].end(), ivf[cid].begin(), ivf[cid].begin() + len);
                    }
                    std::sort(sp_cache[j].begin(), sp_cache[j].end());
                    sp_cache[j].erase(std::unique(sp_cache[j].begin(), sp_cache[j].end()),
                                      sp_cache[j].end());
                    if (sp_cache[j].size() > QP.beamSize)
                    {
                        std::shuffle(sp_cache[j].begin(), sp_cache[j].end(), std::mt19937(0));
                        sp_cache[j].resize(QP.beamSize);
                    }
                    else if (sp_cache[j].size() == 0)
                    {
                        sp_cache[j].push_back(sp);
                    }
                }
            }*/

            return checkRecall(G,
                               Base_Points, Query_Points,
                               Q_Base_Points, Q_Query_Points,
                               GT, k, get_sp, QP, verbose);
        };
        _search_and_parse(G_, G, res_file, k, fixed_beam_width, rerank_factor, rerank_factor, check);
    }
}