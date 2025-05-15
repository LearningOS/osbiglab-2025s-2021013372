### Week 13 Report

#### 本周工作

- 下面是修改代码时阅读代码时认为是重点的部分：

    `llama_kv_cache_unified::state_read`是恢复kv-cache，`llama_kv_cache_unified::state_write`是将kv卸载到内存。

    `llama_context::llama_context`中调用了`llama_kv_cache_unified::init`，在传入参数中`kv_size = cparams.n_ctx`是上下文大小

    在gglm.h的gglm_op中查询相关操作的宏，并且在计算图中填写相关宏，以保证完成相关操作。

    llama-context.cpp中llama_context::kv_self_update函数中调用`graph_compute`进行图计算以及kv更新

    `llm_graph_context::build_attn`会使用`const llama_kv_cache_unified *kv_self`来调用kv cache相关模块

    k,v的长度是256的倍数，而不是实际长度。所以寻找k,v是如何mask的。通过`llm_build_qwen2(const llama_model & model, const llm_graph_params & params, ggml_cgraph * gf) : llm_graph_context(params)`中的`auto * inp_attn = build_attn_inp_kv_unified();`判断了如何进行mask的。因为相关变量会通过`inp=inp_attn`，将这个变量传至`const auto & kq_mask = is_swa ? inp->get_kq_mask_swa() : inp->get_kq_mask();`


- 在ggml.c中只有以下这些操作，没有办法找到对应的index，这其实对变成造成的较大的困扰。但是通过一种特别的方式解决了这个问题（避免手动编写新的算子）

    首先可以在score中寻找top_k，但是不能获得相应位置，所以转换为相应的bool matrix，然后使用bool matrix取乘k和v，就获得了score中top_k的k和v。

    比如`[5, 1, 7, 4, 3]`取前二，则先通过top_k获得前三个`[7, 5, 4]`，然后将所有的数减去4，然后把使用clamp截取范围>=0，对于top_k中的数也相应-4，然后取倒数，分别变成`[1, 0, 3, 0, 0]`和`[1/3, 1]`然后进行外积，最后再次使用缩放和clamp保留其中1的值即可。

    所有操作符如下，没有获得index的算子
    ```C++
    static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {
        "none",

        "x",
        "x+y",
        "x+y",
        "view(x,nb,offset)+=y->x",
        "x-y",
        "x*y",
        "x/y",
        "x^2",
        "√x",
        "log(x)",
        "sin(x)",
        "cos(x)",
        "Σx",
        "Σx_k",
        "Σx/n",
        "argmax(x)",
        "count_equal(x)",
        "repeat(x)",
        "repeat_back(x)",
        "concat(x, y)",
        "silu_back(x)",
        "norm(x)",
        "rms_norm(x)",
        "rms_norm_back(x)",
        "group_norm(x)",
        "l2_norm(x)",

        "X*Y",
        "X[i]*Y",
        "X*Y",

        "x*v",
        "y-\\>view(x)",
        "x-\\>y",
        "cont(x)",
        "reshape(x)",
        "view(x)",
        "permute(x)",
        "transpose(x)",
        "get_rows(x)",
        "get_rows_back(x)",
        "diag(x)",
        "diag_mask_inf(x)",
        "diag_mask_zero(x)",
        "soft_max(x)",
        "soft_max_back(x)",
        "rope(x)",
        "rope_back(x)",
        "clamp(x)",
        "conv_transpose_1d(x)",
        "im2col(x)",
        "im2col_back(x)",
        "conv_transpose_2d(x)",
        "pool_1d(x)",
        "pool_2d(x)",
        "pool_2d_back(x)",
        "upscale(x)",
        "pad(x)",
        "pad_reflect_1d(x)",
        "arange(start, stop, step)",
        "timestep_embedding(timesteps, dim, max_period)",
        "argsort(x)",
        "leaky_relu(x)",

        "flash_attn_ext(x)",
        "flash_attn_back(x)",
        "ssm_conv(x)",
        "ssm_scan(x)",
        "win_part(x)",
        "win_unpart(x)",
        "get_rel_pos(x)",
        "add_rel_pos(x)",
        "rwkv_wkv6(k, v, r, tf, td, s)",
        "gated_linear_attn(k, v, q, gate, s)",
        "rwkv_wkv7(r, w, k, v, a, b, s)",

        "unary(x)",

        "f(x)",
        "f(x,y)",

        "custom_f32(x)",
        "custom_f32(x,y)",
        "custom_f32(x,y,z)",

        "custom(x)",
        "custom(x,y)",
        "custom(x,y,z)",

        "cross_entropy_loss(x,y)",
        "cross_entropy_loss_back(x,y)",
        "adamw(x)",
    };
    ```

    而python的infllm中使用pytorch的算子可以直接获得index
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

- 其他部分的代码展示:

    ```C++
    auto k_need_score_num = n_tokens - score_block_size + 1;
    ggml_tensor * k_need_score = ggml_view_3d(ctx0, k,
        n_embd_head_k, k_need_score_num, n_head_kv,
        k->nb[1],
        k->nb[2],
        0);
    ggml_tensor * kq_need_score = ggml_mul_mat(ctx0, k_need_score, q);
    ggml_mul_mat_set_prec(kq_need_score, GGML_PREC_F32);
    ggml_tensor * score = ggml_view_3d(ctx0, kq_need_score,
        k_need_score_num, score_block_size, kq_need_score->ne[2],
        kq_need_score->nb[1]+1,
        kq_need_score->nb[2],
        0);
    score = ggml_view_2d(ctx0, score, k_need_score_num, score->ne[1]*score->ne[2], score->nb[1], 0);
    score = ggml_transpose(ctx0, score);
    score = ggml_mean(ctx0, score);
    score = ggml_transpose(ctx0, score);
    
    ggml_tensor* kv_score = ggml_view_1d(ctx0, kv_self->score_l[il], k_need_score_num, 0);
    ggml_build_forward_expand(gf, ggml_cpy(ctx0, score, kv_score));
    const_cast<llama_kv_cache_unified *>(kv_self)->score_valid_len = k_need_score_num;
    ```


- 根据infllm中的定义，current tokens $X$和evicted token block $B$之间的关系是

    $\text{sim}(X, B) = \sum^{l_X}_{i=1}\sum^{r_k}_{j=1} q_{i+l_P}\cdot k^B_{b_j}$

    但是仔细观察发现后可以注意到可以将上式的两个求和分割$\text{sim}(X, B) = \sum^{l_X}_{i=1}\sum^{r_k}_{j=1} q_{i+l_P}\cdot k^B_{b_j}= \sum^{l_X}_{i=1} q_{i+l_P}\cdot \sum^{r_k}_{j=1}k^B_{b_j}$

    由于上式中的$\sum^{r_k}_{j=1}k^B_{b_j}$可以考虑改成使用representative score加权计算？或许效果更佳


#### 下周工作计划

- 进行layer prefetch的代码书写

- 考虑对infllm进行改进，比如更适合llama.cpp的操作
