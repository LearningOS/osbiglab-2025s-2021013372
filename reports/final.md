## 实验报告

### 实验目的和实验主要内容

在大模型推理过程中，KV Cache（Key-Value Cache）被广泛用于存储中间计算结果，以减少重复计算开销并提升推理效率。然而，随着模型推理上下文长度的增加，KV Cache的规模急剧扩大，导致GPU显存资源紧张，甚至可能引发模型推理性能下降或无法运行的问题。此外，通过对KV Cache的分析发现，模型推理过程中KV Cache的稀疏性显著，即并非所有存储的KV对都需要参与后续计算。这为优化KV Cache的存储和使用提供了可能性。

基于上述背景，本实验探索将KV Cache部分存储在CPU内存中，仅在推理时动态加载必要的KV对到GPU进行计算。通过这种方式，可以在缓解GPU显存压力的同时，尽可能减少对推理延迟的影响。

本实验旨在复现InfLLM的KV Cache管理方案，结合动态加载策略，优化KV Cache的选取和加载过程。具体目标包括：

1. 针对大模型推理场景，KV Cache的分离存储，将部分KV对存储在CPU内存中，以缓解GPU显存紧张问题；
2. 优化KV Cache的动态加载策略，确保在降低显存占用的同时，尽可能减少对推理延迟的影响；
3. 基于llama.cpp框架，实现InfLLM的KV Cache高效分离方案，并验证其在显存占用和推理效率方面的优化效果。

在llama.cpp中复刻的意义在于，首先llama.cpp的运行速度更快。另外llama.cpp提供了很多后端，而且基于cpp能够在不同平台上编译，便于在包括嵌入式设备在内的各类设备上部署。原有的InfLLM是基于pytorch的，跨平台时难度较大，对于很多平台，可能运行pytorch本身会造成较大开销。

### 实验内容

#### 前期准备工作

自行进行了实现环境搭建，这部分消耗了一定时间。

在NVIDIA RTX 4090上使用CUDA 12.8进行实验。

- InfLLM的实验环境安装

    KV cache的offload相关的3篇文献中，InfLLM是给了明确代码的，所以更容易复现结果，以更加方便移植到llama.cpp中。

    不过InfLLM的环境比较老，使用最新版的各个包会出现错误。

    - 复现结果时，出现了如下bug。加入PYTHONPATH=.的环境变量后解决

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 9, in <module>
            from inf_llm.utils import patch_hf, GreedySearch, patch_model_center
        ModuleNotFoundError: No module named 'inf_llm'
        ```

    - 然后又出现了huggingface要求登录的提示

        加入HUGGING_FACE_HUB_TOKEN的环境变量后解决

    - 然后出现如下报错：

        ```
        Traceback (most recent call last):
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 289, in <module>
            model, tokenizer = get_model_and_tokenizer(args.model)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/benchmark/pred.py", line 56, in get_model_and_tokenizer
            model = patch_hf(model, config.type, **config)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的文件夹>/InfLLM/inf_llm/utils/patch.py", line 152, in patch_hf
            hf_rope = model.model.layers[0].self_attn.rotary_emb 
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "<我的家目录>/miniconda3/envs/infllm/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1928, in __getattr__
            raise AttributeError(
        AttributeError: 'MistralAttention' object has no attribute 'rotary_emb'
        ```

        猜测是版本不同导致rotary_emb没有了，于是检索到一个类似问题

        结合 [AttributeError: 'LlamaAttention' object has no attribute 'rotary_emb'](https://github.com/unslothai/unsloth/issues/1443) 中的回答，执行如下代码，改变环境配置

        ```bash
        pip install unsloth && pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps https://github.com/unslothai/unsloth/archive/refs/tags/November-2024.zip
        pip uninstall tokenizers
        pip uninstall transformers
        pip install tokenizers==0.20.3 transformers==4.46.1
        ```

        如此操作，解决了如上报错


- llama.cpp的实验环境安装

    服务器上存在多个版本的nvcc，需要手动指定nvcc编译器。通过以下指令指定
    ```bash
    export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
    ```

    服务器上存在多个GPU计算节点，而家目录只是挂载到每个计算节点的同一位置。主节点上没有GPU，所以只能在计算节点编译，不过没有计算节点的root权限，所以自行编译cmake等工具链。

    - 手动编译cmake
    
        ```
        wget https://github.com/Kitware/CMake/releases/download/v4.0.0/cmake-4.0.0.tar.gz
        tar axf cmake-4.0.0.tar.gz
        cd cmake-4.0.0
        ./bootstrap
        ./configure --prefix={我的安装目录}/cmake
        make
        make install

        export PATH={我的安装目录}/cmake/bin:$PATH
        ```

        但是在上面安装中遇到了缺少OpenSSL的报错，所以需要自行编译OpenSSL，然后设置OPENSSL_ROOT_DIR环境变量后解决

        ```
        wget https://github.com/openssl/openssl/releases/download/openssl-3.4.1/openssl-3.4.1.tar.gz
        tar axf openssl-3.4.1.tar.gz
        cd openssl-3.4.1
        ./config --prefix={我的安装目录}/openssl
        make
        make install

        export OPENSSL_ROOT_DIR={我的安装目录}/openssl
        ```


    - 编译后工具链后，编译llama.cpp的CUDA版本：

        ```
        cmake -B build -DGGML_CUDA=ON
        cmake --build build --config Release -j
        ```

        然后根据https://qwen.readthedocs.io/zh-cn/latest/run_locally/llama.cpp.html，从hugging face上下载了qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf这个模型，通过以下指令在CLI中运行这个模型。

        ```
        ./llama-cli -m qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf \
            -co -cnv -p "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
            -fa -ngl 80 -n 512
        ```

#### InfLLM优化的主要原理

`context_manager.py`中的`ContextManager`是KV cache相关的主要代码。InfLLM中将kv分成init, global, local三组。对于global部分的kv，使用分组的方式组织，并且在每个分组中保留重要的token的kv，同时记录local_score。`ContextManager`以`MemoryUnit`为单位管理张量将kv cache切分为块卸载到CPU上，而GPU上只保留每个块中有代表性的几个k，以作查询。推理时，结合local_score确定相应块的重要程度，决定是否驻留在显存中。

#### llama.cpp代码分析

llama.cpp项目本身非常复杂，核心代码较多。其前后端分离的设计，使得对于新增的异构设备，只需要提供相应接口，并且能够执行对静态图的计算即可，可以有效兼容各种异构设备。并且cpp的代码能够在大多数平台上编译，并且开销更低。cpp中对于内存管理、同步互斥等的支持更多，可以更方便管理硬件资源，尤其是嵌入式设备等硬件资源有限的设备上。

不过llama.cpp的这种设计导致代码量增加，进行后端抽象以及静态计算图造成了debug和代码阅读的一些麻烦，尤其对于没有经验者。

llama.cpp中代码比较多，有几万行代码，网上能搜到一些参考资料，不过经过llama.cpp的几次重构和功能增加，很多网上的资料和我实际看到的接口并不完全一致。有很多内容，需要手动输出，然后结合代码，猜测其含义。

- 前后端交互方式（以CUDA为例，其他类似）

    `ggml/src/ggml-*`中是各种后端实现和一个`CMakeLists.txt`，主要包括算子和前后端数据传输。

    在cuda后端中有详细的实现，不过有些后端只是对cuda进行简单的函数名替换。比如对于hip后端，在`ggml/src/ggml-cuda/vendors/hip.h`中定义了如下宏（片段）

    ```C++
    ...
    #define cublasCreate hipblasCreate
    #define cublasDestroy hipblasDestroy
    #define cublasGemmEx hipblasGemmEx
    ...
    ```

    - 数据传输部分是通过传递一系列函数指针实现的。
    
        比如`ggml/src/ggml-cuda/ggml-cuda.cu`中的
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
        
        在前端中，这个接口的定义在`ggml/src/ggml-backend-impl.h`如下
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

        设计原理类似于把后端当成一种库，然后把各个调用的入口通过这种方式告知前端。

        讲过包装，实现了`ggml_backend_tensor_set`，调用这个函数可以将数据设定到后端的tensor中，将接口的复杂性保留在实现内部（也方便改接口）。

        ```C++
        void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
            GGML_ASSERT(tensor);
            ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

            if (size == 0) {
                return;
            }

            GGML_ASSERT(buf != NULL && "tensor buffer not set");
            GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
            GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

            buf->iface.set_tensor(buf, tensor, data, offset, size);
        }
        ```

    - 后端各种算子的调用和实现

        这种后端算子的实现，感觉和操作系统中的系统调用中的一些设计比较像（不过当然没有特权级的切换）。把后端作为一种类似“操作系统”的支持，然后前端通过特定的调用号来调用相应操作。不同的后端都约定了相同的OP_CODE和调用方式以便前端可忽略后端的复杂性。

        因为静态计算图，不能直接调用后端函数。所以将需要计算的张量所需要的操作通过GGML_OP_CODE的方式记录，

        ```C++
        // available tensor operations:
        enum ggml_op {
            GGML_OP_NONE = 0,

            GGML_OP_DUP,
            GGML_OP_ADD,
            ...
        }
        ```

        然后比如这个算子，通过给OP_CODE和OP_PARAMS记录下所需要进行的调用。

        （不知道操作系统中有无类似的将很多个系统调用记录下来，然后合并进行调用的方式，以减少系统调用开销？）

        ```C++
        struct ggml_tensor * ggml_abs(
                struct ggml_context * ctx,
                struct ggml_tensor  * a) {
            return ggml_unary(ctx, a, GGML_UNARY_OP_ABS);
        }
                
        struct ggml_tensor * ggml_unary(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_unary_op    op) {
            return ggml_unary_impl(ctx, a, op, false);
        }

        static struct ggml_tensor * ggml_unary_impl(
                struct ggml_context * ctx,
                struct ggml_tensor  * a,
                enum ggml_unary_op    op,
                bool                  inplace) {
            GGML_ASSERT(ggml_is_contiguous_1(a));

            struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

            ggml_set_op_params_i32(result, 0, (int32_t) op);

            result->op     = GGML_OP_UNARY;
            result->src[0] = a;

            return result;
        }

        static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
            assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
            ((int32_t *)(tensor->op_params))[i] = value;
        }
        ```

        然后后端`ggml/src/ggml-cuda/ggml-cuda.cu`调用`ggml_cuda_compute_forward`识别所需要进行的计算。
        ```C++
        static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
            switch (dst->op) {
                case GGML_OP_ARGMAX:
                    ggml_cuda_argmax(ctx, dst);
                    break;
                case GGML_OP_COUNT_EQUAL:
                    ggml_cuda_count_equal(ctx, dst);
                    break;
                ...
            }
        }
        ```

#### InfLLM在llama.cpp中的复现

<!-- InfLLM文章中就提到了在llama.cpp中复现，不过并没有实现。这和llama.cpp和InfLLM的底层数据结构和算法设计不同，导致完全复现实验较为困难。 -->

复现时还是遇到了比较多的问题，比如llama.cpp中算子支持不全，有的算子需要特定条件下才能使用，否则会出现数据不连续、类型不一致等报错信息，比如：

```
llama.cpp/ggml/src/ggml.c:2040: GGML_ASSERT(ggml_can_repeat(b, a)) failed
llama.cpp/ggml/src/ggml-cuda/sumrows.cu:33: GGML_ASSERT(ggml_is_contiguous(src0)) failed
```

- 建立静态计算图

    - qwen2对应的应该是`llama_model::build_graph`中应该是调用`LLM_ARCH_QWEN2`的case。需要定位到核心代码如下：

        ```c++
        struct llm_build_qwen2 : public llm_graph_context {
            llm_build_qwen2(const llama_model & model, const llm_graph_params & params, ggml_cgraph * gf) : llm_graph_context(params) {
                ...
                for (int il = 0; il < n_layer; ++il) {
                    ggml_tensor * inpSA = inpL;

                    // norm
                    cur = build_norm(inpL,
                            model.layers[il].attn_norm, NULL,
                            LLM_NORM_RMS, il);
                    cb(cur, "attn_norm", il);

                    // self-attention
                    {
                        // compute Q and K and RoPE them
                        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
                        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
                        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
                        ...
                        cur = build_attn(inp_attn, gf,
                                model.layers[il].wo, model.layers[il].bo,
                                Qcur, Kcur, Vcur, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
                    }
                    ...
                }
                ...
            }
        };
        ```

        然后定位到需要修改的是attention相关的计算图，在这部分中加入representative相关的逻辑（对于evited部分的token的k和后面若干个q做内积取平均，作为r，然后取top_k作为representative tokens）：

        ```C++
        ggml_tensor * llm_graph_context::build_attn(
                llm_graph_input_attn_kv_unified * inp,
                ggml_cgraph * gf,
                ggml_tensor * wo,
                ggml_tensor * wo_b,
                ggml_tensor * q_cur,
                ggml_tensor * k_cur,
                ggml_tensor * v_cur,
                ggml_tensor * kq_b,
                    float     kq_scale,
                    int       il) const {
            // these nodes are added to the graph together so that they are not reordered
            // by doing so, the number of splits in the graph is reduced
            ...
        }
        ```

        
#### 其他

利用实验中的经验。对课题组中的模型训练进行了系统层面的一些优化。（由于尚未发表，所以这里只展示一些代码片段）

- 考虑硬件的差异。

    - 这台服务器上的GPU计算节点使用光纤连接存储集群，带宽较低，影响训练速度。数据预处理是IO密集型任务，模型训练分离是计算密集型任务。数据预处理受带宽影响很大，由于CPU节点使用IB连接，所以将数据预处理转移到CPU并行处理。数据预处理基本是单核运行，所以同时运行的进程很多，在约50个CPU节点上，使用超过3000个进程处理。如果都在一个文件夹下创建文件，可能超过文件系统的并发能力，所以分散在若干文件夹下，减少文件系统压力。加速几百倍，只需要12h完成了 9 TB的数据预处理（处理后4 TB多）

    - 进行大规模训练，训练数据超过 4 TB，超过内存大小。训练数据放在硬盘上，做异步数据加载，可以明显提高速度，减少GPU等待时间（利用pytorch框架中的功能，实现相对简单）

### 总结

通过本次实验，了解了LLM推理时可能面临的问题，以及一些优化要点和量化分析手段，也了解了llama.cpp的框架和设计逻辑。同时对于环境配置和工具链编译等方面也有了一些经验。对于异构加速器的使用

感谢陈渝老师、王拓为助教和郝子胥助教的指导。

本学期我事情较多，在时间安排上也给大家带来了一些困扰，感谢老师、助教以及同学的理解和支持。
