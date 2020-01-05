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

inline void enableDLA(IBuilder* builder, IBuilderConfig* config, int useDLACore, bool allowGPUFallback=true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore 
                      << " on a platform that does not have any DLA cores" << std::endl; 
            assert("Error: use DLA core on a platform that have no DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!builder->getInt8Mode() && !config->getFlag(BuilderFlag::kINT8))
        {
            // User has no requested INT8 mode. 
            // By default run in FP16 mode. FP32 mode is not permitted. 
            builder->setFp16Mode(true);
            config->setFlag(builderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(BuilderFlag::kSTRICT_TYPES);
    }
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

    // Returns a vector of device buffers that you can use directly as 
    //   bindings for the execute and enqueue methods of IExecutionContext. 
    std::vector<void*>& getDeviceBindings() { return mDeviceBindings; }
    const std::vector<void*>& getDeviceBindings() const { return mDeviceBindings; }

    // Returns the device buffer corresponding to tensorName.
    //   returns nullptr if no such tensor can be found. 
    void* getDeviceBuffer(const std::string& tensorName) const { return getBuffer(false, tensorName); }

    // Returns the host buffer corresponding to tensorName.
    //   returns nullptr if no such tensor can be found. 
    void* getHostBuffer(const std::string& tensorName) const { return getBuffer(false, tensorName); }
    
    // Returns the size of the host and device buffers that coorespond to tensorName. 
    //   returns kINVALID_SIZE_VALUE if no such tensor can be found. 
    size_t size(const std::string& tensorName) const 
    {
        int index = mEngine->getBindingIndex(tensorName.c_str()); 
        if (index == -1)
            return kINVALID_SIZE_VALUE; 
        return mManagedBuffers[index]->hostBuffer.nbBytes();
    }

    // Dump host buffer with specified tensorName to ostream. 
    //   prints error message to std::ostream if no such tensor can be found. 
    void dumpBuffer(std::ostream& os, const std::string& tensorName)
    {
        int index = mEngine->getBindingIndex(tensorName.c_str()); 
        if (index == -1)
        {
            os << "Invalid tensor name" << std::endl; 
            return; 
        }
        void* buf = mManagedBuffers[index]->hostBuffer.data();
        size_t bufSize = mManagedBuffers[index]->hostBuffer.nbBytes();
        nvinfer1::Dims bufDims = mEngine->getBindingDimensions(index);
        size_t rowCount = static_cast<size_t>(bufDims.nbDims >= 1 ? bufDims.d[bufDims.nbDims - 1] : mBatchSize);

        os << "[" << mBatchSize; 
        for (int i=0; i<bufDims.nbDims; i++)
        {
            os << ", " << bufDims.d[i];
        }
        os << "]" << std::endl; 
        switch (mEngine->getBindingDataType(index))
        {
        case nvinfer1::DataType::kINT32: print<int32_t>(os, buf, bufSize, rowCount); break; 
        case nvinfer1::DataType::kFLOAT: print<float>(os, buf, bufSize, rowCount); break; 
        case nvinfer1::DataType::kHALF: print<half_float::half>(os, buf, bufSize, rowCount); break; 
        case nvinfer1::DataType::kINT8: assert(0 && "Int8 network-level input and output is not supported."); break; 
        }
    }

    // Templated print function that dumps buffer of arbitrary type to std::ostream. 
    //    rowCount parameter controls how many elements are on each line. 
    //    a rowCount of 1 means that there is only 1 element on each line. 
    template <tyeename T> 
    void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount)
    {
        assert(rowCount != 0);
        assert(bufSize % sizeof(T) == 0); 
        T* typeBuf = static_cast<T*>(buf); 
        size_t numItems = bufSize / sizeof(T); 
        for (int i=0; i<static_cast<int>(numItems); i++)
        {
            // Handle rowCount == 1 case
            if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
            {
                os << typedBuf[i] << std::endl; 
            }
            else if (rowCount == 1)
            {
                os << typedBuf[i];
            }
            // Handle rowCount > 1 case
            else if (i % rowCount == 0)
            {
                os << typedBuf[i]; 
            }
            else if (i % rowCount == rowCount - 1)
            {
                os << " " << typedBuf[i] << std::endl; 
            }
            else 
            {
                os << " " << typedBuf[i]; 
            }
        }
    }

    // Copy the contents of input host buffers to input device buffers synchronousely. 
    void copyInputToDevice() { memcpyBuffers(true, false, false); }
    
    // Copy the contents of output device buffers to output host buffers synchronousely. 
    void copyOutputToHost() { memcpyBuffers(false, true, false); }

    // Copy the contents of input host buffers to input device buffers asynchronousely. 
    void copyInputToDeviceAsync(const cudaStream_t& stream = 0) { memcpyBuffers(true, false, true, stream); }

    // Copy the contents of output device buffers to output host buffers asynchronousely. 
    void copyOutputToHostAsync(const cudaStream_t& stream = 0) { memcpyBuffers(false, true, true, stream); }

    ~BufferManager() = default; 

private:

    void* getBuffer(const bool isHost, const std::string& tensorName) const 
    {
        int index = mEngine->getBindingIndex(tensorName.c_str());
        if (index == -1)
        {
            return nullptr; 
        }
        return (ishost ? mManagedBuffers[index]->hostBuffer.data() : mManagedBuffers[index]->deviceBuffer.data());
    }

    void memcpyBuffers(const bool copyInput, const bool deviceToHost, const bool async, const cudaStream_t& stream = 0)
    {
        for (int i=0; i<mEngine->getNbBindings(); i++)
        {
            void* dstPtr = 
                deviceToHost ? mManagedBuffers[i]->hostBuffer.data() : mManagedBuffers[i]->deviceBuffers.data();
            const void* srcPtr =
                deviceToHost ? mManagedBuffers[i]->deviceBuffer.data() : mManagedBuffers[i]->hostBuffers.data();
            const size_t byteSize = mManagedBuffers[i]->hostBuffer.nbBytes();
            const cudaMemcpKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
            if ((copyInput && mEngine->bindingIsInput(i)) || (!copyInput && !mEngine->bindingIsInput(i)))
            {
                if (async)
                {
                    CHECK(cudaMemcpyAsync(dstPtr, srcPtr, byteSize, memcpyType, stream));
                }
                else
                {
                    CHECK(cudaMemcpy(dstPtr, srcPtr, byteSize, memcpyType));
                }
                
            }
        }
    }

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    int mBatchSize; 
    std::vector<std::unique_ptr<ManagedBuffer>> mManagedBuffers; 
    std::vecotor<void*> mDeviceBindings;
};

// The SampleUffXor class implements the uff Xor example. 
// It creates the network using a uff model. 
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
    bool processInput(const BufferManager& buffers, 
                      const std::string& inputTensorName, int inputFileIdx) const;

    // Verifies that the output is correct and prints it. 
    bool verifyOutput(const BufferManager& buffers, 
                      const std::string& outputTensorName, 
                      int groundTruthDigit) const; 

    std::shared_ptr<nvinfer::ICudaEngine> mEngine {nullptr}; 
    UffSampleParams mParams; 
    nvinfer1:Dims mInputDims; 
    const int kDIGITS {10};
}; // SampleUffXor

// Creates the network, configure the builder and create the network engine. 
// This function creates the Xor network by parsing the Uff model and builds
// the engine that will used to run Xor. 
bool SampleUffXor::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(
        nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false; 
    }
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if (!network)
    {
        return false; 
    }
    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false; 
    }
    auto parser = SampleUniquePtr<nvinfer1::IUffParser>(nvuffparser::createUffParser());
    if (!parser)
    {
        return false; 
    }
    constructNetwork(parser, network);
    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(16_MiB);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);
    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16); 
    }
    if (mParams.int8)
    {
        config-setFlag(BuilderFlag::kINT8);
    }

    enableDLA(builder.get(), config.get(), mParams.dlaCore);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());
    
    if (!mEngine)
    {
        return false; 
    }
    assert(network->getNbInputs() == 1);
    mInputDims = networkInput(0)->getDimensions();
    assert(mInputDims.nbDims == 3);

    return true; 
}

// Use a Uff parser to create network and marks the output layers. 
