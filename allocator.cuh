#pragma once
#include "commons.cuh"
#include <vector>
#include <map>

constexpr const size_t MAP_SIZE = 32 * 1024 * 1024;
constexpr const size_t MAX_FREE = 2;

class MemoryAllocator {
    std::map<void*, int> sizesAlloc;
    std::map<void*, int> sizesFree;
public:
    template<typename T>
    T * alloc(const size_t num)
    {
        size_t sz = num * sizeof(T);
        // for (const auto& p : sizesFree)
        // {
        //     if (p.second >= sz)
        //     {
        //         sizesAlloc[p.first] = p.second;
        //         sizesFree.erase(p.first);
        //         return (T*)p.first;
        //     }
        // }
        // sz = ceil_div(sz, MAP_SIZE) * MAP_SIZE;
        T * ptr;
        CHECK_CUDA(cudaMalloc(&ptr, sz));
        // sizesAlloc[ptr] = sz;
        return ptr;
    }

    void free(void* ptr)
    {
        CHECK_CUDA(cudaFree(ptr));
        // assert(sizesAlloc.count(ptr));
        // sizesFree[ptr] = sizesAlloc[ptr];
        // sizesAlloc.erase(ptr);
        // while (sizesFree.size() > MAX_FREE)
        // {
        //     CHECK_CUDA(cudaFree(sizesFree.begin()->first));
        //     sizesFree.erase(sizesFree.begin());
        // }
    }

    virtual ~MemoryAllocator()
    {
        for (const auto& p : sizesAlloc)
            CHECK_CUDA(cudaFree(p.first));
        for (const auto& p : sizesFree)
            CHECK_CUDA(cudaFree(p.first));
    }
};
