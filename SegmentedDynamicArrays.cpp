#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <random>
#include <immintrin.h>

using namespace std;
using namespace std::chrono;

struct chunk {
    size_t size;
    int* data;
    int minVal;
    int maxVal;
};

class SegmentedDynamicArray {
public:
    vector<chunk> chunks;
    int currentBuffer;
    int currentElements = 0;
    int* buffer;

    vector<pair<int, int>> insertBuffer;
    const int insertBufferLimit = 5000;

    SegmentedDynamicArray(int bufferSize) {
        currentBuffer = bufferSize;
        buffer = new int[currentBuffer];
    }

    ~SegmentedDynamicArray() {
        clearChunks();
        delete[] buffer;
    }

    void flushInsertBuffer() {
        if (insertBuffer.empty()) return;
        sort(insertBuffer.begin(), insertBuffer.end(), [](const auto &a, const auto &b) {
            return a.first > b.first;
        });
        for (const auto &[index, value] : insertBuffer) {
            insertAt(index, value);
        }
        insertBuffer.clear();
    }

    void writeToChunk() {
        chunk c;
        c.size = currentElements;
        c.data = new int[currentElements];
        memcpy(c.data, buffer, currentElements * sizeof(int));
        c.maxVal=c.data[currentElements-1];
        c.minVal=c.data[0];
        chunks.push_back(c);
        currentElements = 0;
    }

    void addInteger(int value) {
        buffer[currentElements++] = value;
        if (currentElements == currentBuffer) {
            writeToChunk();
        }
    }

    void queueInsert(int index, int value) {
        insertBuffer.push_back({index, value});
        if (insertBuffer.size() >= insertBufferLimit) {
            flushInsertBuffer();
        }
    }

    void insertAt(int index, int value) {
        if (currentElements > 0) writeToChunk();

        int accumulated = 0;
        for (int i = 0; i < chunks.size(); ++i) {
            chunk &c = chunks[i];
            if (index <= accumulated + c.size) {
                int localIdx = index - accumulated;
                chunk newChunk;
                newChunk.size = c.size + 1;
                newChunk.data = new int[newChunk.size];
                memcpy(newChunk.data, c.data, localIdx * sizeof(int));
                newChunk.data[localIdx] = value;
                memcpy(newChunk.data + localIdx + 1, c.data + localIdx, (c.size - localIdx) * sizeof(int));
                delete[] c.data;
                c.data = newChunk.data;
                c.size = newChunk.size;
                return;
            }
            accumulated += c.size;
        }
    }

    int getElement(int index) {
        flushInsertBuffer();
        int runningIndex = 0;
        for (auto& c : chunks) {
            if (index < runningIndex + (int)c.size) {
                return c.data[index - runningIndex];
            }
            runningIndex += c.size;
        }
        return -1;
    }

    void clearChunks() {
        for (auto& c : chunks) {
            delete[] c.data;
        }
        chunks.clear();
    }

    int searchSorted(int target) {
        if (currentElements > 0) writeToChunk();
        flushInsertBuffer();
        int left = 0, right = chunks.size() - 1;
        while (left <= right) {
            int mid = (left + right) / 2;
            chunk& c = chunks[mid];
            int first = c.data[0];
            int last = c.data[c.size - 1];
            if (target < first) {
                right = mid - 1;
            } else if (target > last) {
                left = mid + 1;
            } else {
                int l = 0, r = c.size - 1;
                while (l <= r) {
                    int m = (l + r) / 2;
                    if (c.data[m] == target) {
                        return m + getChunkStartIndex(mid);
                    } else if (c.data[m] < target) {
                        l = m + 1;
                    } else {
                        r = m - 1;
                    }
                }
                return -1;
            }
        }
        return -1;
    }


    int getChunkStartIndex(int chunkIndex) {
        int start = 0;
        for (int i = 0; i < chunkIndex; ++i) {
            start += chunks[i].size;
        }
        return start;
    }







bool simdLinearSearch(const int* data, size_t size, int target) {
    __m256i targetVec = _mm256_set1_epi32(target);  // replicate target into 8 lanes

    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + i));
        __m256i cmp = _mm256_cmpeq_epi32(chunk, targetVec);
        int mask = _mm256_movemask_epi8(cmp);
        if (mask != 0) return true;
    }

    // Handle remaining tail elements
    for (; i < size; ++i) {
        if (data[i] == target) return true;
    }

    return false;
}






bool containsSorted2(int target) {
    if (currentElements > 0) writeToChunk(); // flush buffer if needed
    //flushInsertBuffer(); // flush insert queue

    int left = 0, right = chunks.size() - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        const chunk& c = chunks[mid];

        if (target < c.minVal) {
            right = mid - 1;
        } else if (target > c.maxVal) {
            left = mid + 1;
        } else {
            // Prefetch the chunk's data before searching
            //_mm_prefetch(reinterpret_cast<const char*>(c.data), _MM_HINT_T0);

            // Use SIMD linear search for small chunks, binary search for large ones
            if (c.size <= 512) {
                return simdLinearSearch(c.data, c.size, target);
            } else {
                int l = 0, r = c.size - 1;
                while (l <= r) {
                    int m = (l + r) / 2;
                    if (c.data[m] == target) return true;
                    else if (c.data[m] < target) l = m + 1;
                    else r = m - 1;
                }
                return false;
            }
        }
    }

    return false;
}






};





int main() {
    const int DATA_SIZE = 10'000'000;
    const int QUERY_SIZE = 1'000'000;

    // Generate sorted data
    std::vector<int> sortedData(DATA_SIZE);
    for (int i = 0; i < DATA_SIZE; ++i) sortedData[i] = i;

    // Generate search queries: half present, half absent
    std::vector<int> queries;
    for (int i = 0; i < QUERY_SIZE / 2; ++i) queries.push_back(sortedData[rand() % DATA_SIZE]);
    for (int i = 0; i < QUERY_SIZE / 2; ++i) queries.push_back(DATA_SIZE + rand() % DATA_SIZE);
    std::mt19937 rng(std::random_device{}());
    std::shuffle(queries.begin(), queries.end(), rng);

    std::cout << "\n================= SegmentedDynamicArray Test =====================\n";
    SegmentedDynamicArray sda(512);

    auto start = std::chrono::high_resolution_clock::now();
    for (int val : sortedData)
        sda.addInteger(val);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    start = std::chrono::high_resolution_clock::now();
    volatile int foundSDA = 0;
    for (int q : queries) {
        foundSDA += sda.containsSorted2(q);
    }
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    std::cout << "Total found: " << foundSDA << " / " << QUERY_SIZE << "\n";

    std::cout << "\n=================== std::vector Test =====================\n";
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> vec = sortedData; // Copy data to vector
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Insertion (copy) time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

    start = std::chrono::high_resolution_clock::now();
    volatile int foundVec = 0;
    for (int q : queries)
        foundVec += std::binary_search(vec.begin(), vec.end(), q);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Search time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";
    std::cout << "Total found: " << foundVec << " / " << QUERY_SIZE << "\n";

    vec.clear();
    return 0;
}
