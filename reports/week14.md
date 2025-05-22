### Week 14 Report

#### 本周工作

- 阅读代码后发现，实际上在ggml库中有一个函数可以进行用于取前k个key。

    因为想到llm中中的MoE机制也需要取若干专家的操作，所以应该可以实现这一功能。

    llama-graph.cpp中寻找到了相关代码，阅读后发现使用ggml_top_k和ggml_get_rows这两个函数，可以更为直接地实现infllm的功能。

    ```C++
    ggml_tensor * llm_graph_context::build_moe_ffn(
            ggml_tensor * cur,
            ggml_tensor * gate_inp,
            ggml_tensor * up_exps,
            ggml_tensor * gate_exps,
            ggml_tensor * down_exps,
            ggml_tensor * exp_probs_b,
                int64_t   n_expert,
                int64_t   n_expert_used,
        llm_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                float   w_scale,
            llama_expert_gating_func_type gating_op,
                    int   il) const {
        ...
        // select experts
        ggml_tensor * selected_experts = ggml_top_k(ctx0, selection_probs, n_expert_used); // [n_expert_used, n_tokens]
        cb(selected_experts->src[0], "ffn_moe_argsort", il);
        cb(selected_experts, "ffn_moe_topk", il);

        ggml_tensor * weights = ggml_get_rows(ctx0,
                ggml_reshape_3d(ctx0, probs, 1, n_expert, n_tokens), selected_experts); // [1, n_expert_used, n_tokens]
        cb(weights, "ffn_moe_weights", il);
        ...
    }
    ```

    将这一部分代码重写后，如下所示：
    ```C++
    auto k_need_score_num = n_tokens - score_block_size + 1;
    k_need_score_num = (k_need_score_num/score_block_size)*score_block_size;
    auto k_cur_to_score = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    ggml_tensor * k_need_score = ggml_view_3d(ctx0, k_cur_to_score,
        n_embd_head_k, k_need_score_num, n_head_kv,
        k_cur_to_score->nb[1],
        k_cur_to_score->nb[2],
        0);
    ggml_tensor * kq_need_score = ggml_mul_mat(ctx0, k_need_score, q);
    ggml_mul_mat_set_prec(kq_need_score, GGML_PREC_F32);
    ggml_tensor * score = ggml_view_3d(ctx0, kq_need_score,
        k_need_score_num, score_block_size, kq_need_score->ne[2],
        kq_need_score->nb[1]+ggml_row_size(kq_need_score->type, 1),
        kq_need_score->nb[2],
        0);
    score = ggml_view_2d(ctx0, score, k_need_score_num, score->ne[1]*score->ne[2], score->nb[1], 0);
    score = ggml_transpose(ctx0, score);
    score = ggml_mean(ctx0, score);
    score = ggml_transpose(ctx0, score);
    ggml_tensor* kv_score = ggml_view_1d(ctx0, kv_self->score_l[il], k_need_score_num, 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, score, kv_score));
    const_cast<llama_kv_cache_unified *>(kv_self)->score_valid_len = k_need_score_num;
    
    auto block_score = ggml_view_2d(ctx0, kv_score, score_block_size, k_need_score_num/score_block_size, ggml_row_size(kv_score->type, score_block_size), 0);
    auto block_score_f32 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, block_score->ne[0], block_score->ne[1]);
    block_score_f32 = ggml_cpy(ctx0, block_score, block_score_f32);
    auto score_top_k = ggml_top_k(ctx0, block_score_f32, representative_num);
    ggml_build_forward_expand(gf, score_top_k);
    auto k_represent = ggml_view_4d(ctx0, k_need_score, k_need_score->ne[0], score_block_size, k_need_score->ne[1]/score_block_size, k_need_score->ne[2], k_need_score->nb[1], k_need_score->nb[1]*score_block_size, k_need_score->nb[2], 0);
    auto k_represent_perm = ggml_permute(ctx0, k_represent, 0, 2, 3, 1);
    auto k_represent_top_k = ggml_view_3d(ctx0, k_represent_perm, k_represent_perm->ne[0]*k_represent_perm->ne[1], k_represent_perm->ne[2], k_represent_perm->ne[3], k_represent_perm->nb[2], k_represent_perm->nb[3], 0);
    k_represent_top_k = ggml_get_rows(ctx0, k_represent_top_k, score_top_k);
    k_represent_top_k = ggml_view_4d(ctx0, k_represent_top_k, k_represent_top_k->ne[0]/representative_num, representative_num, k_represent_top_k->ne[1], k_represent_top_k->ne[2], k_represent_top_k->nb[1]/representative_num, k_represent_top_k->nb[1], k_represent_top_k->nb[2], 0);
    ```    

    这一部分代码大致对应python的infllm中使用pytorch的算子可以直接获得index的过程
    ```python
    def get_block_k(self, k, score):
        assert isinstance(score, torch.Tensor)
        assert k.dim() >= 2
        k = self.from_group_kv(k)
        assert k.shape[:-1] == score.shape
        assert k.shape[-2] == self.block_size
        score_topk = score.topk(self.repr_topk, dim=-1).indices
        assert score_topk.shape == (self.num_units, self.unit_size, self.repr_topk)
        ret = torch.gather(k, -2, score_topk[:, :, :, None].expand(self.num_units, self.unit_size, self.repr_topk, self.dim_head))
        return ret
    ```

- 相关代码阅读和debug

    在debug时发现一个奇怪bug:`/public/home/wangyx/file/1099.temp/OS_major_exp/llama.cpp/ggml/src/ggml-cuda/argsort.cu:94: GGML_ASSERT(src0->type == GGML_TYPE_F32) failed`

    而且这个只有在计算的时候才会发生，建立静态计算图时并不会发生这个报错。

    后来阅读代码发现是这样的：
    ```C++
    // ggml_argsort

    struct ggml_tensor * ggml_argsort(
            struct ggml_context  * ctx,
            struct ggml_tensor   * a,
            enum ggml_sort_order   order) {
        GGML_ASSERT(a->ne[0] <= INT32_MAX);
        struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_I32, GGML_MAX_DIMS, a->ne);

        ggml_set_op_params_i32(result, 0, (int32_t) order);

        result->op     = GGML_OP_ARGSORT;
        result->src[0] = a;

        return result;
    }

    // ggml_top_k

    struct ggml_tensor * ggml_top_k(
            struct ggml_context * ctx,
            struct ggml_tensor  * a,
            int                   k) {
        GGML_ASSERT(a->ne[0] >= k);

        struct ggml_tensor * result = ggml_argsort(ctx, a, GGML_SORT_ORDER_DESC);

        result = ggml_view_4d(ctx, result,
                    k, result->ne[1], result->ne[2], result->ne[3],
                    result->nb[1], result->nb[2], result->nb[3],
                    0);

        return result;
    }
    ```

    在ggml_top_k中会调用ggml_argsort进行排序时，在CUDA后端只能支持float32的输入


    ```C++
    void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
        const ggml_tensor * src0 = dst->src[0];
        const float * src0_d = (const float *)src0->data;
        float * dst_d = (float *)dst->data;
        cudaStream_t stream = ctx.stream();

        GGML_ASSERT(src0->type == GGML_TYPE_F32);
        GGML_ASSERT( dst->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src0));

        const int64_t ncols = src0->ne[0];
        const int64_t nrows = ggml_nrows(src0);

        enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

        argsort_f32_i32_cuda(src0_d, (int *)dst_d, ncols, nrows, order, stream);
    }
    ```

    经过查看block_score->type，发现这个张量是GGML_TYPE_F16导致的。
    ```C++
    enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        ...
    }
    ```

    所以阅读其他代码，找到进行转型的方式似乎就是使用ggml_cpy制造一个新的张量，所以插入了这两行代码，从而解决了上述bug。

    ```C++
    auto block_score_f32 = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, block_score->ne[0], block_score->ne[1]);
    block_score_f32 = ggml_cpy(ctx0, block_score, block_score_f32);
    ```

- 对于prefetch相关，和两位助教都讨论了一下，简单阅读了类似功能的仓库，下周开始书写代码。

#### 下周工作计划

- 进行layer prefetch的代码书写
