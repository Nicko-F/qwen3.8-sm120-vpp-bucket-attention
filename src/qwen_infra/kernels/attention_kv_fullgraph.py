def input_key(inputs):
    return tuple(
        (t.data_ptr(), tuple(t.shape), tuple(t.stride()), str(t.dtype), str(t.device)) for t in inputs
    )
