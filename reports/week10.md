### Week 10 Report

#### 本周工作

- 修改代码
    - 首先需要在llama-kv-cache.{h,cpp}中加入关于representative score的变量和修改接口

        所以主要加入了以下变量（具体实现这里不做详细展示）
        `std::vector<ggml_tensor *> llama_kv_cache_unified::score_l;`（每层一个ggml_tensor）
        `void llama_kv_cache_unified::llama_kv_cache_set_attention_score(llama_kv_cache_unified * kv, int layer, int index, float score);`
        `float llama_kv_cache_unified::llama_kv_cache_get_attention_score(const llama_kv_cache_unified * kv, int layer, int index);`

        并且在初始化和更新的时候对representative score进行相应操作

    - 由于llama-kv-cache.cpp中只有kv相关信息，所以需要修改其他代码
    
        llama.cpp和pytorch差别较大，使用静态计算图。需要需要找到相关函数。推理工程的计算图，来获得representative score
        llama-model.{h,cpp}的代码比较多，不过好在根据加载gguf后的输出
        ```
        print_info: arch             = qwen2
        ```
        判断是qwen2对应的`llm_build_qwen2`
        以及在`llama_model::build_graph`中应该是调用`LLM_ARCH_QWEN2`的case。需要定位到核心代码如下：

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

    - 目前似乎还有一点bug，后面debug后展示结果

#### 下周工作计划

- 完成representative score的内容（似乎可以做一些修改，不是严格分块，而是根据语义划分128~256大小不一的块，可能会更好？）

- 做layer by layer的数据预取: issue这个行为和实际操作可能不适合静态图。需要检查依赖，考虑计算图是否被去掉。

- 如果是cpu后端，需要通过ggml_compute_forward调用前后检查相关依赖的方式实现
