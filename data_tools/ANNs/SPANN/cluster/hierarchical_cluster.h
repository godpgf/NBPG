// 随机聚类树
#pragma once
#include <random>
#include <string.h>
#include <functional>
#include <utility>
#include <iostream>
#include <omp.h>
#include <set>
#include <cassert>
#include "avl.h"
#include "log.h"
#include "utils/ivf/ivf.h"
#include "kmeans.h"

namespace ant
{

    template<typename indexType>
    struct RandomIntGen{
        RandomIntGen(indexType sId, indexType eId, size_t seed) : gen(seed), int_dis(sId, eId){

        }

        indexType get_rand_int(){
            return int_dis(gen);
        }

        float get_rand_delta_float(float scale=.0001f){
            auto random_num = get_rand_int();
            if(random_num % 2 == 0)
                random_num *= -1;
            return random_num * scale;
        }

        std::mt19937 gen; 
        std::uniform_int_distribution<indexType> int_dis;
    };

    template <typename indexType>
    struct HierCluster
    {

        struct TreeNode
        {
            TreeNode() : idOffset(0), numElement(0), centroidId(-1), parent(-1) {}
            TreeNode(indexType centroidId, indexType parent) : idOffset(0), numElement(0), centroidId(centroidId), parent(parent) {}

            // 当前聚类的偏移
            indexType idOffset;
            // 当前聚类中包含的元素数量
            indexType numElement;

            std::vector<indexType> childs;
            indexType parent;

            // 质心id，如果当前节点是dummy node，就等于-1。
            indexType centroidId;
        };

        HierCluster(size_t max_vec_num, size_t max_cell_size, uint8_t k, uint max_reps = 8) : max_vec_num(max_vec_num), max_cell_size(max_cell_size), k(k), max_reps(max_reps), cls_tree_avl(max_vec_num), ids_cache(max_vec_num) {}

        // 取样一些随机的点来初始化随机聚类树
        template <typename PointRange>
        void init_hier_cluster(std::vector<indexType>& indices, PointRange &Points, indexType init_vec_num, bool print = true)
        {
            assert(indices.size() > init_vec_num);
            if (init_vec_num < 2)
            {
                std::cout << "init_hier_cluster ERROR: init_vec_num must more then 1!" << std::endl;
                abort();
            }


            // 利用选出来的节点快速初始化一个聚类树
            std::vector<TreeNode> init_tree;
            init_hier_cluster_from_indices(Points, init_tree, indices, init_vec_num, k);

            // 得到每个向量对应的叶子节点，并将叶节点id作为分类cls
            auto vec_num = indices.size();
            
            std::vector<indexType> id2cls;
            id2cls.resize(vec_num);
            size_t max_batch_size = vec_num / 100;
            for (size_t i = 0; i < vec_num; i += max_batch_size)
            {
                size_t cur_batch_size = (i + max_batch_size > vec_num) ? vec_num - i : max_batch_size;

#pragma omp parallel for
                for (size_t j = 0; j < cur_batch_size; ++j)
                {
                    id2cls[i + j] = get_leaf_id(init_tree, Points, Points[i + j], k, i + j);
                }

                if (print)
                    printProgressBar((i + cur_batch_size) / (double)vec_num);
            }

            // 将相同分类的id放到一起去
            std::vector<size_t> nextPos;
            IVF::make_ivf(id2cls, nextPos, ids_cache);

            // 输出聚类树，使用init_tree对cls_tree进行初始化
            size_t all_size = 0;
            std::function<indexType(indexType, indexType)> insert_out_nodes = nullptr;
            insert_out_nodes = [&](indexType from_init_node_id, indexType parent_id) -> size_t
            {
                indexType cur_node_id = cls_tree.size();
                // 将tmp_node_id表示的节点插入到cls_tree中，并返回cls_tree中对应的节点id
                auto &from_node = init_tree[from_init_node_id];                
                cls_tree.push_back(TreeNode(from_node.centroidId, parent_id));

                
                if(from_node.childs.size() == k){
                    // 插入的是分枝节点
                    for(auto cid : from_node.childs){
                        auto new_cid = insert_out_nodes(cid, cur_node_id);
                        
                        auto &cur_node = cls_tree[cur_node_id];
                        assert(cur_node.childs.size() < k);
                        cur_node.childs.push_back(new_cid);
                    }
                } else {
                    auto &cur_node = cls_tree[cur_node_id];
                    // 插入的是叶子节点
                    cur_node.idOffset = nextPos[from_init_node_id - 1];
                    cur_node.numElement = nextPos[from_init_node_id] - cur_node.idOffset;
                    if (cur_node.numElement == 0)
                    {
                        std::cout << "init_hier_cluster ERROR: add empty node!" << std::endl;
                        abort();
                    }
                    cls_tree_avl.insert(cur_node_id, [&](const size_t aId, const size_t bId)
                                        { return cell_comp(aId, bId); });
                }

                return cur_node_id;
            };
            insert_out_nodes(static_cast<indexType>(0), static_cast<indexType>(-1));
        }

        template <typename PointRange>
        void build_hier_cluster(PointRange &Points, size_t seed = 0, bool print = true)
        {
            size_t vec_num = Points.size();
            RandomIntGen<size_t> rand_vec(0, vec_num, seed);
            RandomIntGen<int> rand_dist(1, 100, seed);


            size_t avl_id = cls_tree_avl.get_min();
            auto cur_node_id = cls_tree_avl.get_value(avl_id);
            // 所有孩子的距离
            std::vector<float> allDis(k * cls_tree[cur_node_id].numElement);
            std::vector<std::vector<indexType>> allSplitIndex(k);

            // random cluster
            while (cls_tree_avl.size() < cls_tree_avl.get_max_size() - k && cls_tree[cls_tree_avl.get_value(avl_id)].numElement > this->max_cell_size)
            {
                assert(avl_id != cls_tree_avl.get_max_size());
                auto cur_node_id = cls_tree_avl.get_value(avl_id);
                auto &cur_node = cls_tree[cur_node_id];
                assert(cur_node.childs.size() == 0);
                // 删除当前已经分裂的节点，以便下次不再对其分裂
                cls_tree_avl.erase(avl_id, [&](const size_t aId, const size_t bId)
                                   { return cell_comp(aId, bId); });

                auto *cur_index_ptr = ids_cache.data() + cur_node.idOffset;
                if (cur_node.idOffset + cur_node.numElement > ids_cache.size())
                {
                    std::cout << "build_hier_cluster ERROR: cur_node.idOffset" << cur_node.idOffset << " cur_node.numElement=" << cur_node.numElement << " ids_cache.size()=" << ids_cache.size() << std::endl;
                    abort();
                }

                if (cur_node.numElement == 0)
                {
                    std::cout << "cur_node.numElement == 0!" << std::endl;
                    abort();
                }
                // 随机分裂----------------------------------
                auto new_centroid_ids = cal_centroid_ids(Points, cur_index_ptr, cur_node.numElement, rand_vec);
                assert(new_centroid_ids.size() == k);
                // 插入新节点
                std::vector<TreeNode> new_nodes;
                for(size_t i = 0; i < k; ++i){
                    auto new_cid = new_centroid_ids[i];
                    cur_node.childs.push_back(new_cid);
                    new_nodes.push_back(TreeNode(new_cid, cur_node_id));
                }
                
                // 开始计算距离
                size_t max_batch_size = vec_num / 100;
                for(size_t i = 0; i < cur_node.numElement; i+=max_batch_size){
                    size_t cur_batch_size = (i + max_batch_size > cur_node.numElement) ? cur_node.numElement - i : max_batch_size;
                    #pragma omp parallel for
                    for (size_t j = 0; j < cur_batch_size; ++j)
                    {
                        float* disPtr = allDis.data() + (i + j) * k;
                        for(size_t ki = 0; ki < k; ++ki){
                            auto d = Points[cur_index_ptr[i + j]].distance(Points[new_centroid_ids[ki]]);
                            disPtr[ki] = d;
                        }
                    }
                }

                if(cur_node.numElement <= k){
                    std::cout<<"cur_node.numElement <= k"<<std::endl;
                    abort();
                }

                // 开始写入
                for(int i = 0; i < k; ++i){
                    allSplitIndex[i].resize(0);
                    // 先写入一个元素
                    size_t min_j = 0;
                    float min_d = std::numeric_limits<float>::max();
                    for(size_t j = 0; j < cur_node.numElement; ++j){
                        if(cur_index_ptr[j] == -1)
                            continue;
                        auto d = allDis.data()[j * k + i];
                        if(d <= min_d){
                            min_d = d;
                            min_j = j;
                        }
                    }
                    allSplitIndex[i].push_back(cur_index_ptr[min_j]);
                    cur_index_ptr[min_j] = -1;
                }

                // 先写入一个元素
                for(size_t i = 0; i < cur_node.numElement; ++i){
                    if(cur_index_ptr[i] == -1)
                        continue;
                    float* disPtr = allDis.data() + i * k;
                    float min_dis = std::numeric_limits<float>::max();
                    size_t min_ki = 0;
                    for(size_t j = 0; j < k; ++j){
                        auto d = disPtr[j] + 0.001 * allSplitIndex[j].size();
                        
                        if(d < min_dis){
                            min_dis = d;
                            min_ki = j;
                        }
                    }
                    allSplitIndex[min_ki].push_back(cur_index_ptr[i]);
                }

                size_t cur_idOffset = 0;
                
                for(size_t i = 0; i < k; ++i){
                    new_nodes[i].idOffset = cur_node.idOffset + cur_idOffset;
                    new_nodes[i].numElement = allSplitIndex[i].size();
                    if (allSplitIndex[i].size() == 0 || allSplitIndex[i].size() >= cur_node.numElement)
                    {
                        std::cout << "build_hier_cluster ERROR: split ERROR!"<< allSplitIndex[i].size() << std::endl;
                        abort();
                    }
                    memcpy(cur_index_ptr + cur_idOffset, allSplitIndex[i].data(), allSplitIndex[i].size() * sizeof(indexType));
                    cur_idOffset += allSplitIndex[i].size();
                }

                if(cur_idOffset != cur_idOffset){
                    std::cout<<cur_idOffset<<"!="<<cur_idOffset<<std::endl;
                    abort();
                }
                
                cur_node.idOffset = 0;
                cur_node.numElement = 0;

                auto start_cid = cls_tree.size();
                for(size_t i = 0; i < k; ++i){
                    // delate insert
                    cls_tree.push_back(new_nodes[i]);
                     // 更新平衡二叉树---------------
                    cls_tree_avl.insert(start_cid + i, [&](const size_t aId, const size_t bId)
                                        { return cell_comp(aId, bId); });
                }

                avl_id = cls_tree_avl.get_min();

                if (print)
                    printProgressBar(cls_tree_avl.size() / (double)cls_tree_avl.get_max_size());
                
            }
        }

        void clear()
        {
            cls_tree_avl.clear();
            cls_tree.clear();
            ids_cache.clear();
        }

        // 存储随机聚类树
        std::vector<TreeNode> cls_tree;

        std::vector<indexType> ids_cache;
    protected:
        // 从当前聚类树中得到最接近自己的叶子节点
        template <typename PointRange, typename Point>
        static size_t get_leaf_id(std::vector<TreeNode> &cls_tree, PointRange &Points, Point q, uint8_t k, size_t seed)
        {
            RandomIntGen<int> rig = RandomIntGen<int>(1, 100, seed);
            
            size_t cur_node_id = 0;
            if (cls_tree.size() == 0)
            {
                std::cout << "get_leaf_id ERROR: cls_tree is empty!" << std::endl;
                abort();
            }
            while (cls_tree[cur_node_id].childs.size() == k)
            {
                auto &node = cls_tree[cur_node_id];
                float min_dis = std::numeric_limits<float>::max();
                for(auto cid : node.childs){
                    auto centroidId = cls_tree[cid].centroidId;
                    float d = Points[centroidId].distance(q);
                    if(std::abs(d - min_dis) < 1e-5){
                        d += d * rig.get_rand_delta_float();
                    }
                    
                    if(d < min_dis){
                        min_dis = d;
                        cur_node_id = cid;
                    }
                }
            }
            return cur_node_id;
        }

        // 将随机的向量初始化为随机聚类树的分裂节点
        template <typename PointRange>
        static void init_hier_cluster_from_indices(PointRange &Points, std::vector<TreeNode> &cls_tree, std::vector<indexType>& indices, indexType init_vec_num, uint8_t k)
        {
            if (cls_tree.size() != 0)
            {
                std::cout << "init_hier_cluster_from_indices ERROR: cls_tree must empty!" << std::endl;
                abort();
            }
        
            TreeNode dummyNode;
            cls_tree.push_back(dummyNode);
            for (size_t i = 0; i < init_vec_num; ++i)
            {
                indexType index = indices[i];
                size_t cur_node_id = get_leaf_id(cls_tree, Points, Points[index], k, i);
                if (cls_tree[cur_node_id].childs.size() < k){
                    cls_tree[cur_node_id].childs.push_back(cls_tree.size());
                }

                TreeNode newNode = TreeNode(index, cur_node_id);
                cls_tree.push_back(newNode);
            }
        }

    protected:
        // 得到质心
        template <typename PointRange>
        std::vector<indexType> cal_centroid_ids(PointRange &Points, indexType *indices, indexType indices_size,  RandomIntGen<size_t>& rand_vec){
            std::vector<indexType> centroid_ids;
            if(max_reps == 0){
                std::set<indexType> has_choose;
                size_t cur_indices_size = indices_size;
                while (centroid_ids.size() < k)
                {
                    size_t index = rand_vec.get_rand_int() % cur_indices_size;
                    --cur_indices_size;
                    while (has_choose.find(index) != has_choose.end())
                    {
                        index++;
                    }
                    has_choose.insert(index);
                    centroid_ids.push_back(indices[index]);
                }
            } else {
                train_kmeans(centroid_ids, Points, indices, indices_size, (uint)k);
            }
            
            return centroid_ids;
        }

        int cell_comp(const size_t aId, const size_t bId)
        {
            auto sa = cls_tree[aId].numElement;
            auto sb = cls_tree[bId].numElement;
            if (sa == 0 || sb == 0)
            {
                std::cout << "cell_comp: can't split empty list!" << std::endl;
                abort();
            }

            if (aId >= cls_tree.size() || bId > cls_tree.size())
            {
                std::cout << "cell_comp: Index ERROR!" << std::endl;
                abort();
            }

            if (sa > sb)
            {
                return static_cast<int>(-1);
            }
            if (sa < sb)
            {
                return static_cast<int>(1);
            }
            if (aId < bId)
            {
                return static_cast<int>(-1);
            }
            if (aId > bId)
            {
                return static_cast<int>(1);
            }
            return static_cast<int>(0);
        }

        // 聚类时的向量数量
        size_t max_vec_num;
        // 每个聚类中包含的最大元素数量
        size_t max_cell_size;
        // 每一层随机聚类的簇数量，必须大于等于2，但不能太大
        uint8_t k;

        uint max_reps;

        

        // 聚类时需要选择最长的序列进行分裂，用平衡二叉树来快速获取当前最长序列。
        // 所以，每个节点包含：在聚类树中的节点id（不是当前的平衡二叉树哈）
        AVL<size_t, size_t> cls_tree_avl;
    };

}