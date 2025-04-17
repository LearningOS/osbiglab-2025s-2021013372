### Week 9 Report

#### 本周工作

根据助教建议，阅读有关llama.cpp中find_slot的kv-cache部分代码，而无需过分关心cuda具体实现细节。

`llama_kv_cache_unified::state_read`中恢复kv cache时，首先分配缓冲区，调用`state_read_meta`先对cells进行一些检查和重组，然后使用`state_read_data`读入数据

    - `state_read_meta`

        读取数据的元信息，通过find_slot，使用find_slot在cells中查找是否能够找到足够的

        - `find_slot`中使用了链表的方式组织缓存信息。具体而言，每个cell都使用tail指向下一个的cell，最后一个后面就是空的cell

            在cells中似乎可以同时储存多个sequence，每个sequence的开始是cells[seq_id]，根据tail确定下一个cell（后继的内容）。

            find_slot中还有一步重组的过程，这一过程会将不同的cell的存储位置进行一些交换，以保证算法的高效。（不会出现长短差距过大的情况）

            ```C++
            if (!find_slot(batch)) {
                LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
                return false;
            }
            ```

    - `state_read_data`

        一行就是一个cell，依次将每一层的数据从io缓冲中填充到cuda tensor中。

InfLLM方面，根据助教指导，以及最近学习了CUDA相关技术。

实际上在InfLLM具有一定GPU异步的功能，使用cuda.Event的事件，使得数据加载的操作成为异步。只有获取数据时才使用event.wait()确认事件完成。
但是在传输速度不足时，event.wait()仍会引发等待。

#### 下周工作计划

通过时间测试，发现传输耗时较长，引发GPU进行等待（PCIe总线传输速度不高）
传输的tensor可以考虑进行各种压缩，减少传输量，增加计算量。通过重计算的方式平衡传输和计算时间，缩短关键路径长度。

