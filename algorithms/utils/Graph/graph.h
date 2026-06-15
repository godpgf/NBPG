#pragma once
#include <omp.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <cassert>
#include <cstring>

namespace ant
{
  // 单个节点的邻接表视图；edges[0] 存当前度数，edges[1..] 存邻居 id
  template <typename IndexType>
  struct EdgeRange
  {
    size_t size() const { return edges[0]; }

    IndexType id() const { return node_id_; }

    EdgeRange() : edges(nullptr) {}

    EdgeRange(IndexType *start, IndexType *end, IndexType node_id)
        : edges(start), node_id_(node_id)
    {
      max_degree_ = static_cast<unsigned>((end - start) - 1);
    }

    IndexType operator[](size_t neighbor_idx) const
    {
      if (neighbor_idx >= edges[0])
      {
        std::cout << "g[] ERROR: index exceeds degree while accessing neighbors "
                  << neighbor_idx << ">=" << edges[0] << std::endl;
        abort();
      }
      return edges[neighbor_idx + 1];
    }

    void append_neighbor(IndexType neighbor_id)
    {
      if (edges[0] == max_degree_)
      {
        std::cout << "ERROR in append_neighbor: cannot exceed max degree "
                  << max_degree_ << std::endl;
        abort();
      }
      edges[edges[0] + 1] = neighbor_id;
      edges[0] += 1;
    }

    template <typename RangeType>
    void update_neighbors(const RangeType &neighbors)
    {
      if (neighbors.size() > max_degree_)
      {
        std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                  << max_degree_ << std::endl;
        abort();
      }
      edges[0] = neighbors.size();
      for (size_t i = 0; i < neighbors.size(); ++i)
      {
        edges[i + 1] = neighbors[i];
      }
    }

    template <typename IdDist>
    void update_neighbors(const IdDist *neighbors, IndexType num_neighbors)
    {
      if (num_neighbors > max_degree_)
      {
        std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                  << max_degree_ << std::endl;
        abort();
      }
      edges[0] = num_neighbors;
      for (IndexType i = 0; i < num_neighbors; ++i)
      {
        edges[i + 1] = neighbors[i].index;
      }
    }

    void clear_neighbors() { edges[0] = 0; }

    void prefetch() const
    {
      int cache_lines = static_cast<int>(((edges[0] + 1) * sizeof(IndexType)) / 64);
      for (int i = 0; i < cache_lines; ++i)
      {
        __builtin_prefetch(reinterpret_cast<char *>(edges) + i * 64);
      }
    }

    template <typename Compare>
    void sort(Compare &&compare)
    {
      std::sort(begin(), end(), compare);
    }

    IndexType *begin() const { return edges + 1; }

    IndexType *end() const { return edges + 1 + edges[0]; }

    IndexType *data() const { return edges; }

  private:
    IndexType *edges;
    unsigned max_degree_;
    IndexType node_id_;
  };

  // 图结构：每个节点占用 (max_degree + 1) 个 IndexType
  template <typename IndexType_>
  struct Graph
  {
    using IndexType = IndexType_;

    unsigned max_degree() const { return max_degree_; }
    size_t size() const { return num_nodes_; }

    void resize(size_t new_size)
    {
      if (new_size <= capacity_size_)
      {
        num_nodes_ = new_size;
      }
      else
      {
        std::cout << "graph resize error!" << std::endl;
        abort();
      }
    }

    Graph() {}

    void allocate_graph(unsigned max_degree, size_t num_nodes, bool use_madvise)
    {
      capacity_size_ = num_nodes;
      max_degree_ = max_degree;
      size_t entry_count = num_nodes * (max_degree_ + 1);
      size_t num_bytes = entry_count * sizeof(IndexType);
      IndexType *ptr;
      if (use_madvise)
      {
        ptr = static_cast<IndexType *>(aligned_alloc(1l << 21, num_bytes));
        madvise(ptr, num_bytes, MADV_HUGEPAGE);
      }
      else
      {
        ptr = static_cast<IndexType *>(malloc(num_bytes));
      }
      memset(ptr, 0, num_bytes);

      graph_ = std::shared_ptr<IndexType[]>(ptr, std::free);
    }

    Graph(unsigned max_degree, size_t num_nodes, bool use_madvise = true) : max_degree_(max_degree), num_nodes_(num_nodes)
    {
      allocate_graph(max_degree, num_nodes, use_madvise);
    }

    // 从二进制文件构造图；格式：[num_nodes][max_degree][degrees...][neighbors...]
    Graph(const char *graph_file, bool use_madvise = true)
    {
      std::ifstream reader(graph_file);
      if (!reader.is_open())
      {
        std::cout << "graph file " << graph_file << " not found" << std::endl;
        abort();
      }

      IndexType num_nodes;
      IndexType max_degree;
      reader.read(reinterpret_cast<char *>(&num_nodes), sizeof(IndexType));
      num_nodes_ = num_nodes;
      reader.read(reinterpret_cast<char *>(&max_degree), sizeof(IndexType));
      max_degree_ = max_degree;
      std::cout << "Graph: detected " << num_nodes
                << " points with max degree " << max_degree << std::endl;

      allocate_graph(max_degree_, num_nodes_, use_madvise);

      std::vector<IndexType> degrees(num_nodes_);
      reader.read(reinterpret_cast<char *>(degrees.data()), sizeof(IndexType) * num_nodes_);

      constexpr size_t BLOCK_SIZE = 1000000;
      std::vector<IndexType> neighbor_buffer(BLOCK_SIZE);
      size_t batch_neighbor_count = 0;
      size_t batch_start_node = 0;
      IndexType *graph_data = graph_.get();
      for (size_t i = 0; i < num_nodes_ + 1; ++i)
      {
        if (i == num_nodes_ || batch_neighbor_count + degrees[i] > BLOCK_SIZE)
        {
          reader.read(reinterpret_cast<char *>(neighbor_buffer.data()), batch_neighbor_count * sizeof(IndexType));
          size_t offset = 0;
          for (size_t j = batch_start_node; j < i; ++j)
          {
            graph_data[j * (max_degree_ + 1)] = degrees[j];
            memcpy((*this)[j].begin(), neighbor_buffer.data() + offset, degrees[j] * sizeof(IndexType));
            offset += degrees[j];
          }
          assert(offset == batch_neighbor_count);
          batch_start_node = i;
          batch_neighbor_count = 0;
          if (i == num_nodes_)
          {
            break;
          }
        }
        batch_neighbor_count += degrees[i];
      }

      reader.close();
    }

    // 将图数据加载到已分配的同尺寸图中
    void load(const char *graph_file)
    {
      std::ifstream reader(graph_file);
      if (!reader.is_open())
      {
        std::cout << "graph file " << graph_file << " not found" << std::endl;
        abort();
      }

      IndexType num_nodes;
      IndexType max_degree;
      reader.read(reinterpret_cast<char *>(&num_nodes), sizeof(IndexType));
      reader.read(reinterpret_cast<char *>(&max_degree), sizeof(IndexType));
      if (num_nodes != num_nodes_ || max_degree_ != max_degree)
      {
        std::cout << "graph size error!!!" << std::endl;
        abort();
      }
      num_nodes_ = num_nodes;
      max_degree_ = max_degree;
      std::cout << "Graph: detected " << num_nodes
                << " points with max degree " << max_degree << std::endl;

      std::vector<IndexType> degrees(num_nodes_);
      reader.read(reinterpret_cast<char *>(degrees.data()), sizeof(IndexType) * num_nodes_);

      constexpr size_t BLOCK_SIZE = 1000000;
      std::vector<IndexType> neighbor_buffer(BLOCK_SIZE);
      size_t batch_neighbor_count = 0;
      size_t batch_start_node = 0;
      IndexType *graph_data = graph_.get();
      for (size_t i = 0; i < num_nodes_ + 1; ++i)
      {
        if (i == num_nodes_ || batch_neighbor_count + degrees[i] > BLOCK_SIZE)
        {
          reader.read(reinterpret_cast<char *>(neighbor_buffer.data()), batch_neighbor_count * sizeof(IndexType));
          size_t offset = 0;
          for (size_t j = batch_start_node; j < i; ++j)
          {
            graph_data[j * (max_degree_ + 1)] = degrees[j];
            memcpy((*this)[j].begin(), neighbor_buffer.data() + offset, degrees[j] * sizeof(IndexType));
            offset += degrees[j];
          }
          assert(offset == batch_neighbor_count);
          batch_start_node = i;
          batch_neighbor_count = 0;
          if (i == num_nodes_)
          {
            break;
          }
        }
        batch_neighbor_count += degrees[i];
      }

      reader.close();
    }

    void clear()
    {
#pragma omp parallel for
      for (size_t i = 0; i < num_nodes_; ++i)
      {
        (*this)[i].clear_neighbors();
      }
    }

    // 在 [start_id, start_id + len) 范围内随机初始化邻居
    void random_init(unsigned neighbors_per_node, size_t start_id, size_t len)
    {
      assert(neighbors_per_node < max_degree_);

#pragma omp parallel
      {
        unsigned int seed = omp_get_thread_num();

#pragma omp for
        for (size_t i = start_id; i < start_id + len; ++i)
        {
          auto node = (*this)[i];
          node.clear_neighbors();

          unsigned added = 0;
          unsigned attempts = 0;
          const unsigned max_attempts = neighbors_per_node * 10;

          while (added < neighbors_per_node && attempts < max_attempts)
          {
            IndexType neighbor_id = rand_r(&seed) % len + start_id;
            ++attempts;

            if (neighbor_id == i)
            {
              continue;
            }

            bool exists = false;
            for (size_t j = 0; j < node.size(); ++j)
            {
              if (node[j] == neighbor_id)
              {
                exists = true;
                break;
              }
            }

            if (!exists)
            {
              node.append_neighbor(neighbor_id);
              ++added;
            }
          }
        }
      }
    }

    void save(const char *output_file)
    {
      std::cout << "Writing graph with " << num_nodes_
                << " points and max degree " << max_degree_
                << std::endl;
      std::ofstream writer;
      writer.open(output_file, std::ios::binary | std::ios::out);
      IndexType num_nodes = num_nodes_;
      IndexType max_degree = max_degree_;
      writer.write(reinterpret_cast<char *>(&num_nodes), sizeof(IndexType));
      writer.write(reinterpret_cast<char *>(&max_degree), sizeof(IndexType));

      std::vector<IndexType> degrees(num_nodes_);
#pragma omp parallel for
      for (size_t i = 0; i < num_nodes_; ++i)
      {
        degrees[i] = (*this)[i].size();
      }
      writer.write(reinterpret_cast<char *>(degrees.data()), degrees.size() * sizeof(IndexType));

      constexpr size_t BLOCK_SIZE = 1000000;
      std::vector<IndexType> neighbor_buffer(BLOCK_SIZE);
      size_t batch_neighbor_count = 0;
      for (size_t i = 0; i < num_nodes_ + 1; ++i)
      {
        if (i == num_nodes_ || batch_neighbor_count + degrees[i] > BLOCK_SIZE)
        {
          writer.write(reinterpret_cast<char *>(neighbor_buffer.data()), batch_neighbor_count * sizeof(IndexType));
          batch_neighbor_count = 0;
          if (i == num_nodes_)
          {
            break;
          }
        }

        memcpy(neighbor_buffer.data() + batch_neighbor_count, (*this)[i].begin(), degrees[i] * sizeof(IndexType));
        batch_neighbor_count += degrees[i];
      }

      writer.close();
    }

    EdgeRange<IndexType> operator[](size_t node_id) const
    {
      if (node_id >= num_nodes_)
      {
        std::cout << "ERROR: graph index out of range: " << node_id << std::endl;
        abort();
      }
      return EdgeRange<IndexType>(graph_.get() + node_id * (max_degree_ + 1),
                                  graph_.get() + (node_id + 1) * (max_degree_ + 1),
                                  node_id);
    }

    ~Graph() {}

  private:
    size_t num_nodes_;
    size_t capacity_size_;
    unsigned max_degree_;
    std::shared_ptr<IndexType[]> graph_;
  };

}
