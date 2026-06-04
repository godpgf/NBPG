#pragma once
#include <memory>
#include <omp.h>
#include <iostream>
#include <cassert>

namespace ant
{

    template <typename indexType, typename T>
    struct AVL
    {
        AVL(indexType max_size) : max_size(max_size), cur_size(0), rootId(max_size), nodes(std::shared_ptr<Node[]>(nullptr, std::free)), freeNodeIds(std::shared_ptr<indexType[]>(nullptr, std::free))
        {
            Node *node_ptr = (Node *)malloc(max_size * sizeof(Node));
            indexType *id_ptr = (indexType *)malloc(max_size * sizeof(indexType));
            nodes = std::shared_ptr<Node[]>(node_ptr, std::free);
            freeNodeIds = std::shared_ptr<indexType[]>(id_ptr, std::free);
            clear();
        }

        void clear()
        {
            rootId = max_size;
            cur_size = 0;

#pragma omp parallel for
            for (size_t i = 0; i < max_size; ++i)
            {
                freeNodeIds.get()[i] = static_cast<indexType>(max_size - 1 - i);
            }
        }

        indexType size() { return cur_size; }

        template <typename F>
        indexType insert(const T &val, F &&comp)
        {
            assert(cur_size < max_size);
            rootId = _insert(rootId, val, comp);
            return freeNodeIds.get()[max_size - cur_size];
        }

        template <typename F>
        void erase(indexType nodeId, F &&comp)
        {
            rootId = _erase(rootId, nodeId, comp);
        }

        indexType get_min()
        {
            if (cur_size == 0)
                return max_size;
            auto cur = rootId;
            Node *nds = nodes.get();
            while (!is_empty_id(nds[cur].lchild))
            {
                cur = nds[cur].lchild;
            }
            return cur;
        }

        T get_value(indexType nodeId)
        {
            Node *nds = nodes.get();
            return nds[nodeId].val;
        }

        size_t get_max_size()
        {
            return max_size;
        }

    private:
        template <typename F>
        indexType _insert(indexType headId, const T &val, F &&comp)
        {
            Node *nds = nodes.get();
            if (headId == max_size)
            {
                headId = pop_free_node();
                nodes.get()[headId].fill(val, max_size);
                return headId;
            }
            if (comp(val, nds[headId].val) > 0)
            {
                nds[headId].rchild = _insert(nds[headId].rchild, val, comp);
            }
            else if (comp(val, nds[headId].val) < 0)
            {
                nds[headId].lchild = _insert(nds[headId].lchild, val, comp);
            }
            else
            {
                throw std::invalid_argument("Cannot insert elements with the same value!");
                return max_size;
            }
            nds[headId].height = 1 + std::max(GetHeight(nds[headId].lchild), GetHeight(nds[headId].rchild));
            int bal = GetHeight(nds[headId].lchild) - GetHeight(nds[headId].rchild);
            if (bal > 1)
            {
                if (comp(val, nds[nds[headId].lchild].val) < 0)
                {
                    return LL_Rotate(headId);
                }
                else if (comp(val, nds[nds[headId].lchild].val) > 0)
                {
                    return LR_Rotate(headId);
                }
            }
            if (bal < -1)
            {
                if (comp(val, nds[nds[headId].rchild].val) > 0)
                {
                    return RR_Rotate(headId);
                }
                else if (comp(val, nds[nds[headId].rchild].val) < 0)
                {
                    return RL_Rotate(headId);
                }
            }
            return headId;
        }

        template <typename F>
        indexType _erase(indexType head, indexType nodeId, F &&comp)
        {
            Node *nds = nodes.get();
            auto val = nds[nodeId].val;
            if (is_empty_id(head))
            {
                return max_size;
            }
            if (comp(val, nds[head].val) < 0)
            {
                nds[head].lchild = _erase(nds[head].lchild, nodeId, comp);
            }
            else if (comp(val, nds[head].val) > 0)
            {
                nds[head].rchild = _erase(nds[head].rchild, nodeId, comp);
            }
            else
            {
                if (head != nodeId)
                {
                    std::cout << "error!(" << head << "," << nodeId << ")" << std::endl;
                }
                if (head != nodeId)
                {
                    throw std::invalid_argument("del error!");
                }
                if (is_empty_id(nds[head].lchild) && is_empty_id(nds[head].rchild))
                { // 无左子树无右子树
                    push_free_node(head);
                    head = max_size;
                }
                else if (!is_empty_id(nds[head].lchild) && is_empty_id(nds[head].rchild))
                { // 有左子树无右子树
                    auto lc = nds[head].lchild;
                    push_free_node(head);
                    head = lc;
                }
                else if (is_empty_id(nds[head].lchild) && !is_empty_id(nds[head].rchild))
                { // 无左子树有右子树
                    auto rc = nds[head].rchild;
                    push_free_node(head);
                    head = rc;
                }
                else
                { // 都有
                    auto cur = nds[head].rchild;
                    while (!is_empty_id(nds[cur].lchild))
                    {
                        cur = nds[cur].lchild;
                    }
                    nds[head].val = nds[cur].val;
                    nds[head].rchild = _erase(nds[head].rchild, cur, comp);
                }
            }
            if (is_empty_id(head))
                return head;
            int bal = GetHeight(nds[head].lchild) - GetHeight(nds[head].rchild);
            nds[head].height = 1 + std::max(GetHeight(nds[head].lchild), GetHeight(nds[head].rchild));
            if (bal > 1)
            {
                if (GetHeight(nds[nds[head].lchild].lchild) >= GetHeight(nds[nds[head].lchild].rchild))
                {
                    return LL_Rotate(head);
                }
                else
                {
                    return LR_Rotate(head);
                }
            }
            else if (bal < -1)
            {
                if (GetHeight(nds[nds[head].rchild].lchild) >= GetHeight(nds[nds[head].rchild].rchild))
                {
                    return RL_Rotate(head);
                }
                else
                {
                    return RR_Rotate(head);
                }
            }
            return head;
        }

        indexType pop_free_node()
        {
            cur_size++;
            if (cur_size > max_size)
                std::cout << "cur_size=" << cur_size << " error!" << std::endl;
            assert(cur_size <= max_size);

            return freeNodeIds.get()[max_size - cur_size];
        }

        void push_free_node(indexType nodeId)
        {
            if (cur_size == 0)
                std::cout << "cur_size=" << cur_size << " error!" << std::endl;
            assert(cur_size > 0);
            freeNodeIds.get()[max_size - cur_size] = nodeId;
            cur_size--;
        }

        bool is_empty_id(indexType nodeId) const
        {
            return nodeId == max_size;
        }

        int GetHeight(indexType nodeId) const
        {
            if (is_empty_id(nodeId))
            {
                return 0;
            }
            return nodes.get()[nodeId].height;
        }

        indexType L_Rotate(indexType head)
        {
            Node *nds = nodes.get();
            auto new_head = nds[head].rchild;
            nds[head].rchild = nds[new_head].lchild;
            nds[new_head].lchild = head;
            nds[head].height = 1 + std::max(GetHeight(nds[head].lchild), GetHeight(nds[head].rchild));
            nds[new_head].height = 1 + std::max(GetHeight(nds[new_head].lchild), GetHeight(nds[new_head].rchild));
            return new_head;
        }

        indexType R_Rotate(indexType head)
        {
            Node *nds = nodes.get();
            auto new_head = nds[head].lchild;
            nds[head].lchild = nds[new_head].rchild;
            nds[new_head].rchild = head;
            nds[head].height = 1 + std::max(GetHeight(nds[head].lchild), GetHeight(nds[head].rchild));
            nds[new_head].height = 1 + std::max(GetHeight(nds[new_head].lchild), GetHeight(nds[new_head].rchild));
            return new_head;
        }

        indexType LL_Rotate(indexType head)
        {
            return R_Rotate(head);
        }

        indexType RR_Rotate(indexType head)
        {
            return L_Rotate(head);
        }

        indexType LR_Rotate(indexType head)
        {
            Node *nds = nodes.get();
            auto new_lhead = L_Rotate(nds[head].lchild);
            nds[head].lchild = new_lhead;
            return R_Rotate(head);
        }

        indexType RL_Rotate(indexType head)
        {
            Node *nds = nodes.get();
            auto new_rhead = R_Rotate(nds[head].rchild);
            nds[head].rchild = new_rhead;
            return L_Rotate(head);
        }

    private:
        struct Node
        {
            T val;
            indexType height;
            indexType lchild;
            indexType rchild;

            void fill(const T &_val, indexType empty_id)
            {
                val = _val;
                height = 1;
                lchild = empty_id;
                rchild = empty_id;
            }
        };
        std::shared_ptr<Node[]> nodes;
        std::shared_ptr<indexType[]> freeNodeIds;

        indexType max_size;
        indexType cur_size;
        indexType rootId;
    };
    void test_avl()
    {
        AVL<int, float> avl(100);
        auto comp = [&](float a, float b)
        {
            if (a < b)
                return static_cast<int>(-1);
            if (a > b)
                return static_cast<int>(1);
            return static_cast<int>(0);
        };

        for (int i = 0; i < 100; ++i)
        {
            float v = ((size_t)std::rand()) % 1000000 + 1;
            avl.insert(v / 1000.f, comp);
        }

        for (int i = 0; i < 100; ++i)
        {
            int id = avl.get_min();
            std::cout << avl.get_value(id) << " ";
            avl.erase(id, comp);
        }
        std::cout << std::endl;
    }
}
