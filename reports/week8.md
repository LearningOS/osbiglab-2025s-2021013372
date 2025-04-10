# Week 8 Report (mid-term)

这几周研究了InfLLM和llama.cpp这两个项目中的kv-cache方式，并且对比两者的不同，以便将InfLLM和llama.cpp进行结合以提升性能。

## InfLLM代码结构分析及其kv-cache的方式

InfLLM中将kv分成init, global, local三组。对于global部分的kv，使用分组的方式组织，并且在每个分组中保留重要的token的kv，同时记录local_score。`ContextManager`以`MemoryUnit`为单位管理张量，结合local_score确定kv的重要程度，决定是否驻留在显存中。

不过InfLLM中内存和显存间数据传输都是串行的，没有做到计算和数据传输在时间上的重叠，这导致了kv的加载卸载比较低效。


## llama.cpp代码结构分析及其kv-cache的方式

llama.cpp兼容各种后端，所以对接口进行了抽象。使用`gglm`库屏蔽了后端的复杂性，比如`ggml_tensor`作为统一的接口保存张量的信息，`ggml_backend_buffer_type`作为后端设备和函数接口的抽象。


在`llama-model.{h,cpp}`中，`llama_model::impl`保存张量缓存相关信息，其中`ctxs`中保存张量的基本信息，`bufs`中保存张量的内容。
```C++
struct llama_model {
...
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
```

```C++
struct llama_model::impl {
...
    // contexts where the model tensors metadata is stored
    std::vector<ggml_context_ptr> ctxs;

    // the model memory buffers for the tensor data
    std::vector<ggml_backend_buffer_ptr> bufs;

    buft_list_t cpu_buft_list;
    std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;

    struct layer_dev {
        ggml_backend_dev_t dev;
        buft_list_t * buft_list;
    };

    layer_dev dev_input = {};
    layer_dev dev_output = {};
    std::vector<layer_dev> dev_layer;
...
};
```

在进行推理时，`llama-context.cpp`在创建`llama_context`时，会创建`llama_kv_cache_unified`，然后调用`llama_kv_cache_unified::init`初始化，以对kv进行cache。

在`llama_kv_cache_unified::init`中

- 获取每一层的的后端buffer类型`ggml_backend_buffer_type_t`，然后根据后端buffer类型创建`ggml_context`上下文来，然后使用这个上下文创建`ggml_tensor* k, *v;`，最后进行了张量初始化。

    `ggml_backend_buffer_type::iface`中保存了后端buffer相关的接口。
    ```C++
    struct ggml_backend_buffer_type {
        struct ggml_backend_buffer_type_i  iface;
        ggml_backend_dev_t device;
        void * context;
    };
    ```

    对于CUDA而言，`ggml_backend_cuda_buffer_type_alloc_buffer`给出了CUDA中分配buffer的接口，其中会先调用`ggml_cuda_device_malloc`分配显存，然后包装成`ggml_backend_cuda_buffer_context`，最后经过`ggml_backend_buffer_init`返回统一的`ggml_backend_buffer`后端buffer类的指针。

    ```C++
    static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
        /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
        /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
        /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
        /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
        /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
        /* .is_host          = */ NULL,
    };
    ```

`llama_kv_cache_unified`

`cells`的长度为`kv_size`，这个参数来自 `params.n_ctx`，表示最长的上下文长度。

`llama_kv_cache_unified::state_write`->`llama_io_write_buffer::write_tensor`->`ggml_backend_tensor_get`->`ggml_backend_buffer_i::get_tensor`->`gglm-cuda.cu`中的`ggml_backend_cuda_buffer_get_tensor`通过这种调用关系将tensor写入`llama_io_write_buffer`。基于类似的调用方式，`llama_kv_cache_unified::state_read`实现从`llama_io_write_buffer`读取数据到tensor。


总体而言，kv-cache的方式比较naive，没有长输入的优化，所以加入InfLLM可以提高性能。
