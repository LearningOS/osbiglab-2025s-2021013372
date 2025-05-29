### Week 15 Report

#### 本周工作

- 由于涉及到CUDA device和host之间的数据传递，所以需要先调研在ggml库中如何实现的。

    经过代码阅读和分析，找到`ggml/src/ggml-cuda/ggml-cuda.cu`中的
    ```C++
    static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

        ggml_cuda_set_device(ctx->device);
        CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }
    ```

    然后通过下面这个结构体，将函数入口传递到ggml_backend_cuda_buffer_interface的接口中
    ```C++
    static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
        /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
        /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
        /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
        /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_cuda_buffer_clear,
        /* .reset           = */ NULL,
    };
    ```

    由于前后端分离，这个接口的定义在`ggml/src/ggml-backend-impl.h`如下
    ```C++
    //
    // Backend buffer
    //

    struct ggml_backend_buffer_i {
        // (optional) free the buffer
        void         (*free_buffer)  (ggml_backend_buffer_t buffer);
        // base address of the buffer
        void *       (*get_base)     (ggml_backend_buffer_t buffer);
        // (optional) initialize a tensor in the buffer (eg. add tensor extras)
        enum ggml_status (*init_tensor)(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor);
        // tensor data access
        void         (*memset_tensor)(ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor,     uint8_t value, size_t offset, size_t size);
        void         (*set_tensor)   (ggml_backend_buffer_t buffer,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void         (*get_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        // (optional) tensor copy: dst is in the buffer, src may be in any buffer, including buffers from a different backend (return false if not supported)
        bool         (*cpy_tensor)   (ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst);
        // clear the entire buffer
        void         (*clear)        (ggml_backend_buffer_t buffer, uint8_t value);
        // (optional) reset any internal state due to tensor initialization, such as tensor extras
        void         (*reset)        (ggml_backend_buffer_t buffer);
    };
    ```
    看起来我们可以通过set_tensor和get_tensor两个函数就可以实现前后端分离的张量加载卸载。

    于是分析各个类之间的包装关系：
    ```C++
    struct ggml_backend_buffer {
        struct ggml_backend_buffer_i  iface;
        ggml_backend_buffer_type_t    buft;
        void * context;
        size_t size;
        enum ggml_backend_buffer_usage usage;
    };
    ```

    通过ggml_backend_buffer_init
    ```C++
    // backend buffer

    ggml_backend_buffer_t ggml_backend_buffer_init(
                ggml_backend_buffer_type_t buft,
            struct ggml_backend_buffer_i      iface,
                void *                     context,
                size_t                     size) {
        ggml_backend_buffer_t buffer = new ggml_backend_buffer {
            /* .interface = */ iface,
            /* .buft      = */ buft,
            /* .context   = */ context,
            /* .size      = */ size,
            /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
        };

        return buffer;
    }
    ```

    在llm_graph_context中有一个sched，
    ```C++
    struct llm_graph_context {
        ...
        ggml_backend_sched * sched;

        ggml_backend * backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?
        ...
    }
    ```

    而sched的类型是ggml_backend_sched，ggml_backend_sched::backends可以访问都这个后端接口。
    ```C++
    struct ggml_backend_sched {
        bool is_reset; // true if the scheduler has been reset since the last graph split
        bool is_alloc;

        int n_backends;

        ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
        ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
        ggml_gallocr_t galloc;
        ...
    }
    ```

    而backend_cpu的类型是ggml_backend，ggml_backend::iface可以访问都这个后端接口。
    ```C++
    struct ggml_backend_sched {
        bool is_reset; // true if the scheduler has been reset since the last graph split
        bool is_alloc;

        int n_backends;

        ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
        ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
        ggml_gallocr_t galloc;
        ...
    }
    ```

    分析下面代码，可以知道上述的接口被传入到set_tensor_async，于是使用这一接口进行张量设置。
    ```C++
    typedef struct             ggml_backend * ggml_backend_t;

    ...

    //
    // Backend (stream)
    //

    struct ggml_backend_i {
        const char * (*get_name)(ggml_backend_t backend);

        void (*free)(ggml_backend_t backend);

        // (optional) asynchronous tensor data access
        void (*set_tensor_async)(ggml_backend_t backend,       struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
        void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor,       void * data, size_t offset, size_t size);
        bool (*cpy_tensor_async)(ggml_backend_t backend_src, ggml_backend_t backend_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

        // (optional) complete all pending operations (required if the backend supports async operations)
        void (*synchronize)(ggml_backend_t backend);

        // (optional) graph plans (not used currently)
        // compute graph with a plan
        ggml_backend_graph_plan_t (*graph_plan_create) (ggml_backend_t backend, const struct ggml_cgraph * cgraph);
        void                      (*graph_plan_free)   (ggml_backend_t backend, ggml_backend_graph_plan_t plan);
        // update the plan with a new graph - this should be faster than creating a new plan when the graph has the same topology
        void                      (*graph_plan_update) (ggml_backend_t backend, ggml_backend_graph_plan_t plan, const struct ggml_cgraph * cgraph);
        // compute the graph with the plan
        enum ggml_status          (*graph_plan_compute)(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

        // compute graph (always async if supported by the backend)
        enum ggml_status          (*graph_compute)     (ggml_backend_t backend, struct ggml_cgraph * cgraph);

        // (optional) event synchronization
        // record an event on this stream
        void (*event_record)(ggml_backend_t backend, ggml_backend_event_t event);
        // wait for an event on on a different stream
        void (*event_wait)  (ggml_backend_t backend, ggml_backend_event_t event);
    };

    struct ggml_backend {
        ggml_guid_t guid;
        struct ggml_backend_i iface;
        ggml_backend_dev_t device;
        void * context;
    };

    ```

    由于llm_build_qwen2继承自llm_graph_context，所以在这个类中可以通过this->backend_cpu->iface.set_tensor_async进行张量设置。
    ```C++
    struct llm_build_qwen2 : public llm_graph_context {
        llm_build_qwen2(const llama_model & model, const llm_graph_params & params, ggml_cgraph * gf) : llm_graph_context(params) {
            ...
        }
    ```

    参考`ggml/src/ggml-backend.cpp`中的这个函数，认为llama.cpp应该是这样使用接口的
    ```C++
    void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
        GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
        GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

        if (backend->iface.set_tensor_async == NULL) {
            ggml_backend_tensor_set(tensor, data, offset, size);
        } else {
            backend->iface.set_tensor_async(backend, tensor, data, offset, size);
        }
    }
    ```

    注意backend_buffer，并且注意同步防止async的问题
