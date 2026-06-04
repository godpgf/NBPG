#pragma once
#include <vector>
#include "utils/mmap.h"

namespace ant
{

  template <typename T>
  struct GroundTruth
  {
    std::vector<T> coords;
    std::vector<float> dists;
    long dim;
    size_t n;

    GroundTruth() {}

    GroundTruth(const char *gtFile, bool has_dist = true)
    {
      if (gtFile == NULL)
      {
        n = 0;
        dim = 0;
      }
      else
      {
        auto [fileptr, length] = mmapStringFromFile(gtFile);

        int num_vectors = *((T *)fileptr);
        int d = *((T *)(fileptr + 4));

        std::cout << "Ground truth: detected " << num_vectors << " points with num results " << d << std::endl;

        T *start_coords = (T *)(fileptr + 8);
        T *end_coords = start_coords + d * num_vectors;
        coords.resize(d * num_vectors);
        memcpy(coords.data(), start_coords, d * num_vectors * sizeof(T));

        n = num_vectors;
        dim = d;

        if (has_dist)
        {
          float *start_dists = (float *)(end_coords);
          dists.resize(d * num_vectors);
          memcpy(dists.data(), start_dists, d * num_vectors * sizeof(float));
        }
      }
    }

    T coordinates(long i, long j) const { return *(coords.begin() + i * dim + j); }

    float distances(long i, long j) const { return *(dists.begin() + i * dim + j); }

    size_t size() const { return n; }

    long dimension() const { return dim; }
  };

}