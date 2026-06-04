#pragma once
#include <algorithm>
#include <set>

namespace ant
{
  template <typename GraphType>
  std::pair<double, uint32_t> graph_stats_(GraphType &G, size_t gSize = 0)
  {
    if (gSize == 0)
      gSize = G.size();
    uint32_t maxDegree = 0;
    size_t sum1 = 0;
    for (size_t i = 0; i < gSize; ++i)
    {
      auto deg = G[i].size();
      sum1 += deg;
      if (deg > maxDegree)
        maxDegree = deg;
    }
    double avg_deg = sum1 / ((double)gSize);
    return std::make_pair(avg_deg, maxDegree);
  }

  template <typename GraphType>
  double graph_stats_(GraphType &G, std::vector<float> &popular, size_t gSize = 0)
  {
    if (gSize == 0)
      gSize = G.size();
    float sum_popular = 0;
    for (auto p : popular)
      sum_popular += p;
    float scale = gSize / sum_popular;
    for (auto i = 0; i < gSize; ++i)
    {
      popular[i] *= scale;
    }

    double sum1 = 0;
    for (size_t i = 0; i < gSize; ++i)
    {
      auto deg = G[i].size();
      sum1 += deg * popular[i];
    }
    double avg_deg = sum1 / ((double)gSize);
    return avg_deg;
  }

  struct Graph_
  {
    std::string name;
    std::string params;
    long size;
    double avg_deg;
    int max_deg;
    double time;
    size_t isolated_cnt;
    double popular_avg_deg;

    Graph_() {}

    Graph_(std::string n, std::string p, long s, double ad, int md, double t, size_t isolated_cnt = 0, double popular_avg_deg=0)
        : name(n), params(p), size(s), avg_deg(ad), max_deg(md), time(t), isolated_cnt(isolated_cnt), popular_avg_deg(popular_avg_deg) {}

    void print()
    {
      std::cout << name << " graph built with " << size
                << " points and parameters " << params << std::endl;
      std::cout << "Graph has average degree " << avg_deg
                << ", popular average degree " << popular_avg_deg
                << " and maximum degree " << max_deg << std::endl;
      std::cout << "Isolated nodes num is " << isolated_cnt << std::endl;
      std::cout << "Graph built in " << time << " seconds" << std::endl;
    }
  };

  struct NN_Result
  {
    double recall;

    uint32_t avg_cmps;
    uint32_t tail_cmps;

    uint32_t avg_visited;
    uint32_t tail_visited;

    float QPS;

    int k;
    int beamQ;
    float cut;
    int limit;
    int degree_limit;
    int gtn;

    long num_queries;

    template <typename indexType>
    NN_Result(double r, std::vector<indexType> stats, float qps, int K, int Q,
              float c, long q, int limit, int degree_limit, int gtn)
        : recall(r),
          QPS(qps),
          k(K),
          beamQ(Q),
          cut(c),
          limit(limit),
          degree_limit(degree_limit),
          gtn(gtn),
          num_queries(q)
    {
      if (stats.size() != 4)
        abort();

      avg_cmps = stats[0];
      tail_cmps = stats[1];
      avg_visited = stats[2];
      tail_visited = stats[3];
    }

    void print()
    {
      std::cout << "For " << gtn << "@" << gtn << " recall = " << recall
                << ", QPS = " << QPS << ", Q = " << beamQ << ", cut = " << cut;
      std::cout << ", visited limit = " << limit << ", degree limit: " << degree_limit;
      std::cout << ", average visited = " << avg_visited << ", average cmps = " << avg_cmps << std::endl;
    }

    void print_verbose()
    {
      std::cout << "Over " << num_queries << " queries" << std::endl;
      std::cout << "k = " << k << ", Q = " << beamQ << ", cut = " << cut
                << ", throughput = " << QPS << "/second" << std::endl;
      std::cout << "Recall: " << recall << std::endl;
      std::cout << "Average dist cmps: " << avg_cmps
                << ", 99th percentile dist cmps: " << tail_cmps << std::endl;
      std::cout << "Average num visited: " << avg_visited
                << ", 99th percentile num visited: " << tail_visited << std::endl;
    }
  };

  template <typename res>
  auto parse_result(std::vector<res> results,
                    std::vector<float> buckets)
  {
    std::vector<float> ret_buckets;
    std::vector<res> retval;
    for (int i = 0; i < buckets.size(); i++)
    {
      float b = buckets[i];
      auto pred = [&](res R)
      { return R.recall >= b; };
      std::vector<res> candidates;

      std::vector<res> temp_candidates;
      std::copy_if(results.begin(), results.end(), std::back_inserter(temp_candidates), pred);

      if ((i == buckets.size() - 1) || (temp_candidates.size() == 0))
      {
        candidates = temp_candidates;
      }
      else
      {
        float c = buckets[i + 1];
        auto pred2 = [&](res R)
        { return R.recall <= c; };
        std::copy_if(temp_candidates.begin(), temp_candidates.end(), std::back_inserter(candidates), pred2);
      }
      if (candidates.size() != 0)
      {
        auto less = [&](res R, res S)
        { return R.QPS < S.QPS; };
        res M = *(std::max_element(candidates.begin(), candidates.end(), less));
        //   res M = *(parlay::max_element(candidates, less));
        M.print();
        retval.push_back(M);
        ret_buckets.push_back(b);
      }
    }
    return std::make_pair(retval, ret_buckets);
  }
}
