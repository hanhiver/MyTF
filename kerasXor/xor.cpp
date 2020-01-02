#include "NvInfer.h"
#include "NvUffParser.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <fstream> 
#include <string> 
#include <sys/stat.h>
#include <unordered_map>
#include <vector> 

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const 
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

struct SampleParams
{
    int batchSize {1};
    int dlaCore {-1}; 
    bool int8 {false};
    bool fp16 {false};
    std::vector<std::string> dataDirs; 
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames; 
};

struct UffSampleParams : public SampleParams
{
    std::string uffFileName; 
};

inline int64_t volume(const nvinfer1:Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

template <typename AllocFunc, typename FreeFunc>
class GenericBuffer
{
public:
    GenericBuffer(nvinfer1::DataType type = nvinfer1::dataType::kFLOAT)
        : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr)
    {
    }

    GenericBuffer(size_t size, nvinfer1::DataType type)
        : mSize(size), mCapacity(size), mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes()))
        {
            throw std::bad_alloc {};
        }
    }

    // for std::move()
    GenericBuffer(GenericBuffer&& buf)
        : mSize(buf.mSize), mCapacity(buf.mCapacity), mType(buf.mType), mBuffer(buf.mBuffer)
    {
        buf.mSize = 0; 
        buf.mCapacity = 0; 
        buf.mType = nvinfer1::DataType::kFLOAT; 
        buf.mBuffer = nullptr; 
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf)
        {
            freeFn(mBuffer); 
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType; 
            mBuffer = buf.mBuffer; 

            buf.mSize = 0; 
            buf.mCapacity = 0; 
            buf.mBuffer = nullptr; 
        }
        return *this; 
    }

    size_t size() const
    {
        return mSize; 
    }

    size_t nbBytes() const 
    {
        return this->size() * getElementSize(mType);
    }

    void resize(size_t newSize)
    {
        mSize = newSize; 
        if (mCapacity < newSize)
        {
            freeFn(mBuffer);
            if (!allocFn(&mBuffer, this->nbBytes()))
            {
                throw std::bad_alloc {}; 
            }
            mCapacity = newSize;
        }
    }

    void resize(const nvinfer1::Dims& dims)
    {
        return this->resize(volume(dims)); 
    }

    ~GenericBuffer()
    {
        freeFn(mBuffer);
    }

private: 
    size_t mSize {0}, mCapacity {0};
    nvinfer1::DataType mType; 
    void* mBuffer; 
    AllocFunc allocFn; 
    FreeFunc freeFn; 
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>; 

class ManagedBuffer
{
public:
    DeviceBuffer deviceBuffer; 
    HostBuffer hostBuffer; 
};

class BufferManager
{
public: 
    static const size_t kINVALID_SIZE_VALUE = ~size_t(0); 

    // Create a BufferManager for handling buffer interactions with engine. 
    BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, 
                  const int& batchSize, 
                  const nvinfer1::IExecutionContext* context = nullptr)
        : mEngine(engine), mBatchSize(batchSize)
    {
        // Create host and device buffers. 
        for (int i=0; i<mEngine->getNbBindings(); i++)
        {
            auto dims = context ? context->getBindingDimensions(i) : mEngine->getBindingDimensions(i);
            size_t vol = context ? 1 : static_cast<size_t>(mBatchSize);
            nvinfer1::DataType type = mEngine->getBindingDataType(i);
            int vecDim = mEngine->getBindingVectorizedDim(i);
            if (-1 != vecDim)
            {
                int scalarPerVec = mEngine->getBindingComponentsPerElement(i);
                dims.d[vecDim] = divUp(dims.d[vecDim], scalarsPerVec);
                vol *= scalarsPervec; 
            }
            vol *= volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf {new ManagedBuffer()}; 
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type); 
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            mManagedBuffers.emplace_back(std::move(manBuf)); 
        }
    }

    // Return a vector of device buffers that you can use directly as 
    //   bindings for the execute and enqueue methods of IExecutionContext. 
    std::vector<void*>& getDeviceBindings() { return mDeviceBindings; }

    buffers.h : 303 lines. 

};

class SampleUffXor
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>; 

public: 
    SampleUffXor(const UffSampleParams& params) 
        : mParams(params)
    {
    }
    
    bool build();      // Builds the network engine.  
    bool infer();      // Runs the TensorRT inference engine. 
    bool teardown();   // Used to clean up any state created in the class. 

private: 
    // Parses a Uff model and creates a TensorRT network. 
    void constructNetwork(SampleUniquePtr<nvuffparser::IUffParser>& parser, 
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network);
    
    // Reads the input and mean data, preprocesses, and stores the result
    //    in a managed buffer. 
    bool processInput(XXX); 
}