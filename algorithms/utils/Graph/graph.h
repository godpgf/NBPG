#pragma once
#include <omp.h>
#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <cassert>

namespace ant
{

  template <typename indexType>
  struct edgeRange
  {

    size_t size() const { return edges[0]; }

    indexType id() const { return id_; }

    edgeRange() : edges(nullptr) {}

    edgeRange(indexType *start, indexType *end, indexType id)
        : edges(start), id_(id)
    {
      maxDeg = (end - start) - 1;
    }

    indexType operator[](size_t j) const
    {
      if (j > edges[0])
      {
        std::cout << "g[] ERROR: index exceeds degree while accessing neighbors " << j << ">" << edges[0] << std::endl;
        abort();
      }
      else
        return edges[j + 1];
    }

    void append_neighbor(indexType nbh)
    {
      if (edges[0] == maxDeg)
      {
        std::cout << "ERROR in append_neighbor: cannot exceed max degree "
                  << maxDeg << std::endl;
        abort();
      }
      else
      {
        edges[edges[0] + 1] = nbh;
        edges[0] += 1;
      }
    }

    template <typename rangeType>
    void update_neighbors(const rangeType &r)
    {
      if (r.size() > maxDeg)
      {
        std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                  << maxDeg << std::endl;
        abort();
      }
      edges[0] = r.size();
      for (int i = 0; i < r.size(); i++)
      {
        edges[i + 1] = r[i];
      }
    }

    template <typename id_dist>
    void update_neighbors(const id_dist *nbhs, indexType num_nbh)
    {
      if (num_nbh > maxDeg)
      {
        std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                  << maxDeg << std::endl;
        abort();
      }
      edges[0] = num_nbh;
      for (int i = 0; i < num_nbh; i++)
      {
        edges[i + 1] = nbhs[i].index;
      }
    }

    void clear_neighbors()
    {
      edges[0] = 0;
    }

    void prefetch() const
    {
      int l = ((edges[0] + 1) * sizeof(indexType)) / 64;
      for (int i = 0; i < l; i++)
        __builtin_prefetch((char *)edges + i * 64);
    }

    template <typename F>
    void sort(F &&less)
    {
      std::sort(this->begin(), this->end(), less);
    }

    indexType *begin() const { return edges + 1; }

    indexType *end() const { return edges + 1 + edges[0]; }

    indexType *data() const { return edges; }

  private:
    indexType *edges;
    unsigned maxDeg;
    indexType id_;
  };

  template <typename indexType_>
  struct Graph
  {
    using indexType = indexType_;

    unsigned max_degree() const { return maxDeg; }
    size_t size() const { return n; }
    void resize(size_t new_size)
    {
      if (new_size <= capacity_size)
      {
        n = new_size;
      }
      else
      {
        std::cout << "graph resize error!" << std::endl;
        abort();
      }
    }

    Graph() {}

    void allocate_graph(unsigned maxDeg, size_t n, bool use_madvise)
    {
      capacity_size = n;
      size_t cnt = n * (maxDeg + 1);
      size_t num_bytes = cnt * sizeof(indexType);
      indexType *ptr;
      if (use_madvise)
      {
        ptr = (indexType *)aligned_alloc(1l << 21, num_bytes);
        madvise(ptr, num_bytes, MADV_HUGEPAGE);
      }
      else
      {
        ptr = (indexType *)malloc(num_bytes);
      }
      memset(ptr, 0, num_bytes);

      graph = std::shared_ptr<indexType[]>(ptr, std::free);
    }

    Graph(unsigned maxDeg, size_t n, bool use_madvise = true) : maxDeg(maxDeg), n(n)
    {
      allocate_graph(maxDeg, n, use_madvise);
    }

    Graph(const char *gFile, bool use_madvise = true)
    {
      std::ifstream reader(gFile);
      if (!reader.is_open())
      {
        std::cout << "graph file " << gFile << " not found" << std::endl;
        abort();
      }

      // read num points and max degree
      indexType num_points;
      indexType max_deg;
      reader.read((char *)(&num_points), sizeof(indexType));
      n = num_points;
      reader.read((char *)(&max_deg), sizeof(indexType));
      maxDeg = max_deg;
      std::cout << "Graph: detected " << num_points
                << " points with max degree " << max_deg << std::endl;

      allocate_graph(max_deg, n, use_madvise);

      // 先读出每个节点邻居的数量
      std::vector<indexType> sizes = std::vector<indexType>(n);
      reader.read((char *)sizes.data(), sizeof(indexType) * n);

      size_t BLOCK_SIZE = 1000000;
      std::vector<indexType> data = std::vector<indexType>(BLOCK_SIZE);
      size_t cur_size = 0;
      size_t cur_index = 0;
      indexType *gr = graph.get();
      for (size_t i = 0; i < n + 1; ++i)
      {
        if (i == n || cur_size + sizes[i] > BLOCK_SIZE)
        {
          // 读出数据
          reader.read((char *)data.data(), cur_size * sizeof(indexType));
          size_t cur_offset = 0;
          for (size_t j = cur_index; j < i; ++j)
          {
            gr[j * (maxDeg + 1)] = sizes[j];
            memcpy((*this)[j].begin(), data.data() + cur_offset, sizes[j] * sizeof(indexType));
            cur_offset += sizes[j];
          }
          assert(cur_offset == cur_size);
          cur_index = i;
          cur_size = 0;
          if (i == n)
            break;
        }
        cur_size += sizes[i];
      }

      reader.close();
    }

    void load(const char *gFile)
    {
      std::ifstream reader(gFile);
      if (!reader.is_open())
      {
        std::cout << "graph file " << gFile << " not found" << std::endl;
        abort();
      }

      // read num points and max degree
      indexType num_points;
      indexType max_deg;
      reader.read((char *)(&num_points), sizeof(indexType));

      reader.read((char *)(&max_deg), sizeof(indexType));
      if (num_points != n || maxDeg != max_deg)
      {
        std::cout << "graph size error!!!" << std::endl;
        abort();
      }
      n = num_points;
      maxDeg = max_deg;
      std::cout << "Graph: detected " << num_points
                << " points with max degree " << max_deg << std::endl;

      // 先读出每个节点邻居的数量
      std::vector<indexType> sizes = std::vector<indexType>(n);
      reader.read((char *)sizes.data(), sizeof(indexType) * n);

      size_t BLOCK_SIZE = 1000000;
      std::vector<indexType> data = std::vector<indexType>(BLOCK_SIZE);
      size_t cur_size = 0;
      size_t cur_index = 0;
      indexType *gr = graph.get();
      for (size_t i = 0; i < n + 1; ++i)
      {
        if (i == n || cur_size + sizes[i] > BLOCK_SIZE)
        {
          // 读出数据
          reader.read((char *)data.data(), cur_size * sizeof(indexType));
          size_t cur_offset = 0;
          for (size_t j = cur_index; j < i; ++j)
          {
            gr[j * (maxDeg + 1)] = sizes[j];
            memcpy((*this)[j].begin(), data.data() + cur_offset, sizes[j] * sizeof(indexType));
            cur_offset += sizes[j];
          }
          assert(cur_offset == cur_size);
          cur_index = i;
          cur_size = 0;
          if (i == n)
            break;
        }
        cur_size += sizes[i];
      }

      reader.close();
    }

    void clear()
    {
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i)
      {
        (*this)[i].clear_neighbors();
      }
    }

    void random_init(unsigned neighbors, size_t start_id, size_t len)
    {
      assert(neighbors < maxDeg);
      // 随机初始化每个节点的邻居，如果随机初始化后，有某个邻居是当前节点，就把这个邻居删除

      // 使用 OpenMP 并行处理每个节点
      #pragma omp parallel
      {
        // 每个线程使用自己的随机数生成器，避免竞争
        unsigned int seed = omp_get_thread_num();

        #pragma omp for
        for (size_t i = start_id; i < start_id + len; ++i)
        {
          auto node = (*this)[i];
          node.clear_neighbors();

          // 尝试添加 neighbors 个邻居
          unsigned added = 0;
          unsigned attempts = 0;
          const unsigned max_attempts = neighbors * 10; // 防止无限循环

          while (added < neighbors && attempts < max_attempts)
          {
            // 生成随机邻居 ID [0, n)
            indexType nbh = rand_r(&seed) % len + start_id;
            attempts++;

            // 跳过自身
            if (nbh == i)
            {
              continue;
            }

            // 检查是否已存在（避免重复邻居）
            bool exists = false;
            for (size_t j = 0; j < node.size(); ++j)
            {
              if (node[j] == nbh)
              {
                exists = true;
                break;
              }
            }

            if (!exists)
            {
              node.append_neighbor(nbh);
              added++;
            }
          }
        }
      }
    }

    void save(const char *oFile)
    {
      std::cout << "Writing graph with " << n
                << " points and max degree " << maxDeg
                << std::endl;
      std::ofstream writer;
      writer.open(oFile, std::ios::binary | std::ios::out);
      indexType num_points = n;
      indexType max_deg = maxDeg;
      writer.write((char *)&num_points, sizeof(indexType));
      writer.write((char *)&max_deg, sizeof(indexType));

      // 先存每个节点的邻居数量
      std::vector<indexType> sizes = std::vector<indexType>(n);
#pragma omp parallel for
      for (size_t i = 0; i < n; ++i)
      {
        sizes[i] = (*this)[i].size();
      }
      writer.write((char *)sizes.data(), sizes.size() * sizeof(indexType));

      size_t BLOCK_SIZE = 1000000;
      std::vector<indexType> data = std::vector<indexType>(BLOCK_SIZE);
      size_t cur_size = 0;
      for (size_t i = 0; i < n + 1; ++i)
      {
        if (i == n || cur_size + sizes[i] > BLOCK_SIZE)
        {
          // 写入之前的数据
          writer.write((char *)data.data(), cur_size * sizeof(indexType));
          cur_size = 0;
          if (i == n)
            break;
        }

        memcpy(data.data() + cur_size, (*this)[i].begin(), sizes[i] * sizeof(indexType));
        cur_size += sizes[i];
      }

      writer.close();
    }

    edgeRange<indexType> operator[](size_t i) const
    {
      if (i > n)
      {
        std::cout << "ERROR: graph index out of range: " << i << std::endl;
        abort();
      }
      return edgeRange<indexType>(graph.get() + i * (maxDeg + 1),
                                  graph.get() + (i + 1) * (maxDeg + 1),
                                  i);
    }

    ~Graph() {}

  private:
    size_t n;
    size_t capacity_size;
    unsigned maxDeg;
    std::shared_ptr<indexType[]> graph;
  };

} // end namespace
