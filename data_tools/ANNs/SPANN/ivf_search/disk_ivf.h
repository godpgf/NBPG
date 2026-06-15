#pragma once
#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <set>
#include <functional>
#include "utils/ivf/ivf.h"
#include "utils/mmap.h"

namespace ant
{
    template <typename ivfDataType>
    struct IVFWriter
    {
        IVFWriter(std::string ivf_path, size_t keyNum, size_t valueNum, size_t max_cache_size = 100000)
            : ivf_path(ivf_path),
              keyNum(keyNum),
              valueNum(valueNum),
              max_cache_size(max_cache_size),
              nextPos(keyNum),
              cur_index(0)
        {
            if (ivf_path != "")
            {
                init_writer();
            }
        }

        template <typename rangeType>
        size_t fill(const rangeType &r)
        {
            if (nextPos.size() < cur_index)
            {
                std::cout << "IVFWrite ERROR: nextPos.size() < cur_index" << std::endl;
                abort();
            }
            auto l = r.size() + (cur_index == 0 ? 0 : nextPos[cur_index - 1]);
            if (cur_index == nextPos.size())
            {
                nextPos.push_back(l);
            }
            else
            {
                nextPos[cur_index] = l;
            }

            for (auto d : r)
                ivf.push_back(d);
            if (ivf_path != "" && ivf.size() > max_cache_size)
            {
                writer.write((char *)ivf.data(), ivf.size() * sizeof(ivfDataType));
                ivf.clear();
            }
            return cur_index++;
        }

        size_t fill(const ivfDataType *begin, const ivfDataType *end)
        {
            if (nextPos.size() < cur_index)
            {
                std::cout << "IVFWrite ERROR: nextPos.size() < cur_index" << std::endl;
                abort();
            }
            size_t data_size = (end - begin);
            auto l = data_size + (cur_index == 0 ? 0 : nextPos[cur_index - 1]);
            if (cur_index == nextPos.size())
            {
                nextPos.push_back(l);
            }
            else
            {
                nextPos[cur_index] = l;
            }

            const ivfDataType *p = begin;
            while (p != end)
            {
                ivf.push_back(*p);
                p++;
            }

            if (ivf_path != "" && ivf.size() > max_cache_size)
            {
                writer.write((char *)ivf.data(), ivf.size() * sizeof(ivfDataType));
                ivf.clear();
            }
            return cur_index++;
        }

        void finish()
        {
            if (ivf_path != "")
            {
                if (ivf.size() > 0)
                {
                    writer.write((char *)ivf.data(), ivf.size() * sizeof(ivfDataType));
                    ivf.clear();
                }
                writer.close();

                if (keyNum == 0)
                {
                    keyNum = nextPos.size();
                    writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out);
                    writer.write((char *)&keyNum, sizeof(size_t));
                    writer.write((char *)&valueNum, sizeof(size_t));
                    writer.write((char *)nextPos.data(), nextPos.size() * sizeof(size_t));
                    // 当前是动态添加的数据
                    std::ifstream reader;
                    reader.open((ivf_path + ".tmp").c_str(), std::ios::binary | std::ios::in);
                    ivf.resize(max_cache_size);

                    for (size_t i = 0; i < nextPos[keyNum - 1]; i += max_cache_size)
                    {
                        size_t cur_batch_size = (i + max_cache_size > nextPos[keyNum - 1]) ? nextPos[keyNum - 1] - i : max_cache_size;
                        reader.read((char *)ivf.data(), cur_batch_size * sizeof(ivfDataType));
                        writer.write((char *)ivf.data(), cur_batch_size * sizeof(ivfDataType));
                    }
                }
                else
                {
                    // 重新写入
                    writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out | std::ios::in);
                    writer.seekp(sizeof(size_t) * 2, std::ios::beg);
                    writer.write((char *)nextPos.data(), nextPos.size() * sizeof(size_t));
                }

                writer.close();
            }
        }

        std::ofstream writer;
        std::string ivf_path;
        size_t keyNum;
        size_t valueNum;
        // 最多缓存多少ivf，当超过时就存盘
        size_t max_cache_size;
        // 记录当前写入的是第几个元素
        size_t cur_index;

        std::vector<size_t> nextPos;
        std::vector<ivfDataType> ivf;

    protected:
        void init_writer()
        {
            if (nextPos.size() == 0)
            {
                // 动态添加数据
                writer.open((ivf_path + ".tmp").c_str(), std::ios::binary | std::ios::out);
            }
            else
            {
                writer.open(ivf_path.c_str(), std::ios::binary | std::ios::out);
                writer.write((char *)&keyNum, sizeof(size_t));
                writer.write((char *)&valueNum, sizeof(size_t));
                // 先占个位置
                writer.write((char *)nextPos.data(), keyNum * sizeof(size_t));
            }
        }
    };

    template <typename indexType>
    struct IVFReader
    {
        IVFReader(std::string ivf_path = "", bool use_cache = true) : ivf_path(ivf_path), use_cache(use_cache), cache_values(std::shared_ptr<indexType[]>(nullptr, std::free)) //, mmapContext(ivf_path.c_str())
        {

            if (ivf_path != "")
            {
                std::ifstream reader;
                reader.open(ivf_path.c_str(), std::ios::in | std::ios::binary);
                if (!reader.is_open())
                {
                    std::cout << "Data file " << ivf_path << " not found" << std::endl;
                    std::abort();
                }
                reader.read((char *)&keyNum, sizeof(size_t));
                reader.read((char *)&valueNum, sizeof(size_t));
                nextPos.resize(keyNum);
                head_offset = (2 + nextPos.size()) * sizeof(size_t);
                reader.read((char *)nextPos.data(), nextPos.size() * sizeof(size_t));
                if (use_cache)
                {
                    num_workers = 1;
                    // 一次加载所有数据
                    indexType *ptr = (indexType *)malloc(nextPos[keyNum - 1] * sizeof(indexType));
                    reader.read((char *)ptr, nextPos[keyNum - 1] * sizeof(indexType));
                    cache_values = std::shared_ptr<indexType[]>(ptr, std::free);
                }
                else
                {
                    num_workers = omp_get_max_threads();

                    // 申请缓存
                    for (auto i = 1; i < keyNum; ++i)
                        max_element_num = std::max(max_element_num, nextPos[i] - nextPos[i - 1]);
                    indexType *ptr = (indexType *)malloc(num_workers * max_element_num * sizeof(indexType));
                    cache_values = std::shared_ptr<indexType[]>(ptr, std::free);

                    // 文件读取-----------------
                    for (auto i = 0; i < num_workers; ++i)
                    {
                        std::ifstream cur_reader;
                        cur_reader.open(ivf_path.c_str(), std::ios::in | std::ios::binary);
                        reader_list.push_back(std::move(cur_reader));
                    }
                }
                reader.close();
            }
            else
            {
                std::cout << "ivf path is empty!" << std::endl;
                abort();
            }
        }

        ~IVFReader()
        {
            for (auto i = 0; i < reader_list.size(); ++i)
                reader_list[i].close();
        }

        size_t size() const { return nextPos.size(); }

        std::pair<indexType *, size_t> operator[](size_t i)
        {
            if (i >= nextPos.size() || i < 0)
            {
                std::cout << "ERROR: point index out of range: " << i << " from range [" << 0 << ", " << nextPos.size() << ")" << std::endl;
                abort();
            }

            size_t pre_pos = (i == 0) ? 0 : nextPos[i - 1];
            size_t data_size = nextPos[i] - pre_pos;

            indexType *to_data = nullptr;
            if (use_cache || data_size == 0)
            {
                to_data = cache_values.get() + pre_pos;
            }
            else
            {
                auto worker_id = omp_get_thread_num() % num_workers;
                size_t offset = head_offset + pre_pos * sizeof(indexType);
                to_data = cache_values.get() + worker_id * max_element_num;
                /*try
                {
                    char *data = mmapContext.map_region(offset, data_size * sizeof(indexType));

                    memcpy(to_data, data, data_size * sizeof(indexType));
                    MMapContext::unmap_region(data, offset, data_size * sizeof(indexType));
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Thread " << worker_id << " error: " << e.what() << std::endl;
                }*/
                reader_list[worker_id].seekg(offset);
                reader_list[worker_id].read((char *)to_data, data_size * sizeof(indexType));
            }
            return std::pair(to_data, data_size);
        }

        size_t head_offset;
        size_t num_workers;
        // MMapContext mmapContext;

        std::string ivf_path;
        size_t keyNum;
        size_t valueNum;

        bool use_cache;
        size_t max_element_num = 0;
        std::vector<size_t> nextPos;
        std::shared_ptr<indexType[]> cache_values;

        std::vector<std::ifstream> reader_list;
    };

    template <typename dataType, typename indexType>
    struct IVFPosReader
    {
        IVFPosReader(const char *filename) : filename(filename)
        {
            std::ifstream ivf_pos_reader;
            ivf_pos_reader.open(filename, std::ios::binary | std::ios::in);
            ivf_pos_reader.read((char *)&keyNum, sizeof(size_t));
            ivf_pos_reader.read((char *)&valueNum, sizeof(size_t));
            ivf_pos_reader.read((char *)&dims, sizeof(size_t));
            nextPos.resize(keyNum);
            ivf_pos_reader.read((char *)nextPos.data(), keyNum * sizeof(size_t));
            ivf_pos_reader.close();
            headOffset = (3 + keyNum) * sizeof(size_t);
            element_bytes = sizeof(indexType) + sizeof(dataType) * dims;
        }

        void init()
        {
            std::tie(fileptr, filelength) = mmapStringFromFile(filename);
        }

        void clear()
        {
            if (filelength > 0)
            {
                if(munmap(fileptr, filelength) == -1){
                    std::cout<<"munmap failed"<<std::endl;
                    abort();
                }
            }
        }

        std::pair<indexType *, dataType *> load_ivf_and_pos(size_t cid)
        {
            if (cid >= nextPos.size())
            {
                std::cout << "cid error!" << std::endl;
                abort();
            }
            size_t pre_pos = (cid == 0) ? 0 : nextPos[cid - 1];
            size_t data_size = nextPos[cid] - pre_pos;
            char *ptr = fileptr + headOffset + pre_pos * element_bytes;
            // 预加载
            prefetch_async(ptr, data_size * element_bytes);
            return std::pair<indexType *, dataType *>(
                (indexType *)ptr,
                (dataType *)(ptr + data_size * sizeof(indexType)));
        }

        size_t keyNum;
        size_t valueNum;
        size_t dims;
        size_t headOffset;
        size_t element_bytes;
        std::vector<size_t> nextPos;
        const char *filename;

        char *fileptr;
        size_t filelength;
    };

    template <typename indexType>
    void split_ivf(IVFReader<indexType> &fromIvf, const std::vector<indexType> &fromCentroidIds, size_t maxIVFLen, IVFWriter<indexType> &toIvf, std::vector<indexType> &toCentroidIds, std::function<void(indexType *, size_t, std::vector<indexType> &, std::vector<indexType> &)> calId2Cls)
    {
        if (toIvf.nextPos.size() > 0 || toIvf.ivf.size() > 0 || toCentroidIds.size() > 0)
        {
            std::cout << "make_ivf:ERROR nextPos or ivf list must empty!" << std::endl;
            abort();
        }

        if (fromIvf.nextPos.size() != fromCentroidIds.size())
        {
            std::cout << "make_ivf:ERROR fromNextPos.size() != fromCentroidIds.size()" << std::endl;
            abort();
        }

        // 记录已经使用过的质心
        std::set<indexType> has_seen;

        size_t max_ivf_size = 0;

        for (size_t i = 0; i < fromIvf.nextPos.size(); ++i)
        {
            size_t pre_offset = i > 0 ? fromIvf.nextPos[i - 1] : 0;
            size_t ivf_size = fromIvf.nextPos[i] - pre_offset;
            if (ivf_size == 0)
                continue;

            auto [ivf_data, data_size] = fromIvf[i];
            assert(data_size == ivf_size);

            if (data_size > max_ivf_size)
                max_ivf_size = data_size;

            // if (ivf_size < maxIVFLen && has_seen.find(fromCentroidIds[i]) == has_seen.end())
            if (true)
            {
                // 如果当前的ivf没有超长
                has_seen.insert(fromCentroidIds[i]);
                toCentroidIds.push_back(fromCentroidIds[i]);
                std::vector<indexType> vec(ivf_data, ivf_data + ivf_size);
                toIvf.fill(vec);
            }
            else
            {
                // ivf超长，那么丢弃之前的质心，开始随机分裂！！！

                // 1、随机选择split_num个质心
                std::vector<indexType> randCentroidIds;
                for (size_t j = 0; j < ivf_size; ++j)
                {
                    if (has_seen.find(ivf_data[j]) == has_seen.end())
                        randCentroidIds.push_back(ivf_data[j]);
                }

                // 所有质心都被选择过，将丢弃这个质心的ivf
                if (randCentroidIds.size() == 0)
                    continue;

                // 随机选出split_num个新质心
                uint32_t split_num = ivf_size / maxIVFLen;
                if (split_num > randCentroidIds.size())
                    split_num = randCentroidIds.size();
                if (split_num == 0)
                    split_num = 1;

                if (split_num < randCentroidIds.size())
                {
                    std::mt19937_64 rng{std::random_device{}()};                       // 64 位引擎
                    std::shuffle(randCentroidIds.begin(), randCentroidIds.end(), rng); // 洗牌
                    randCentroidIds.resize(split_num);                                 // 只保留前 split_num 个
                }

                // 预加载数据
                // std::vector<typename PQPR::byte> quant_bytes = std::vector<typename PQPR::byte>(split_num * quants.get_dims());
                // quants.load2cache(randCentroidIds.data(), randCentroidIds.size());
                // parlay::parallel_for(0, randCentroidIds.size(), [&](size_t ii)
                //  { memcpy(quant_bytes.data() + ii * quants.get_dims(), quants[randCentroidIds[ii]].data(), quants.get_dims() * sizeof(typename PQPR::byte)); });

                // Points.load2cache(ivf_data, data_size);

                // 记录ivf中第几个元素所对应的“新质心的下标”，下面称为id
                std::vector<indexType> id2cls = std::vector<indexType>(ivf_size, 0);
                /*parlay::parallel_for(0, ivf_size, [&](size_t ii)
                                     {
                        // size_t indexOffset = pre_offset + ii;
                        float minDis = std::numeric_limits<float>::max();
                        size_t min_j;

                        for(size_t j = 0; j < split_num; ++j){
                            //indexType centroidIndex = randCentroidIds[j];
                            //auto* qnt = quants[centroidIndex].data();
                            auto* qnt = quant_bytes.data() + j * quants.get_dims();
                            float dis = pivot.pq_distance_sum(Points[ivf_data[ii]].data(), qnt);
                            // float dis = Points[ivf_data[ii]].pq_distance_sum(pivot.pivots.data(), pivot.order_ids.data(), pivot.times, pivot.chunk, qnt);
                            if(dis < minDis){
                                minDis = dis;
                                min_j = j;
                            }
                        }
                        id2cls[ii] = min_j; });*/

                // tmp_ivf记录每个随机质心包含的id，0<=id<ivf_size
                std::vector<size_t> tmp_nextPos;
                std::vector<indexType> tmp_ivf;
                calId2Cls(ivf_data, data_size, randCentroidIds, id2cls);
                IVF::make_ivf(id2cls, tmp_nextPos, tmp_ivf);

                // 3、写入结果
                /*if(tmp_nextPos.size() != split_num){
                    std::cout<<"tmp_nextPos.size() ERROR! tmp_nextPos.size()="<<tmp_nextPos.size()<<" split_num="<<split_num<<std::endl;
                    abort();
                }*/
                for (size_t j = 0; j < tmp_nextPos.size(); ++j)
                {
                    // 遍历所有新质心
                    size_t tmp_pre_offset = j > 0 ? tmp_nextPos[j - 1] : 0;
                    size_t tmp_ivf_size = tmp_nextPos[j] - tmp_pre_offset;
                    if (tmp_ivf_size == 0)
                        continue;
                    // 插入新质心和它包含的元素数量
                    // has_seen.insert(fromCentroidIds[i]);
                    has_seen.insert(randCentroidIds[j]);
                    toCentroidIds.push_back(randCentroidIds[j]);

                    // 插入新质心包含的元素
                    std::vector<indexType> vs;
                    for (auto jj = 0; jj < tmp_ivf_size; ++jj)
                    {
                        auto cur_id = tmp_ivf[tmp_pre_offset + jj];
                        vs.push_back(ivf_data[cur_id]);
                    }
                    toIvf.fill(vs);
                }
            }
        }

        std::cout << "max_ivf_size:" << max_ivf_size << std::endl;

        toIvf.finish();
    }



}