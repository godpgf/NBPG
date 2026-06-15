#pragma once
#include "utils/Graph/dfs.h"

namespace ant
{
    // HNSW 分层图沿用统一 DFS 实现，保留 HierDFS 别名以兼容旧代码
    using HierDFS = DFS;
}
