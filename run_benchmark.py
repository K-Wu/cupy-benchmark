from benchmarks.cupy import bench_core
from benchmarks.cupy import bench_fusion
from benchmarks.cupy import bench_linalg
import nvtx

if __name__=="__main__":
    for dtype in ['float32', 'complex128']:
        for ndim in [0, 1, 2, 5, 8]:
            for in_order in  ['C', 'F']:
                for out_order in ['C', 'F']:
                    with nvtx.annotate("bench_core.Array({dtype},{ndim},{in_order},{out_order})".format(dtype=dtype,ndim=ndim,in_order=in_order,out_order=out_order), color="purple"):

                        array=bench_core.Array()
                        array.setup(dtype, ndim, in_order, out_order)
                        with nvtx.annotate("bench_core.Array.array_from_numpy({dtype},{ndim},{in_order},{out_order})".format(dtype=dtype,ndim=ndim,in_order=in_order,out_order=out_order), color="green"):
                            array.time_array_from_numpy(dtype, ndim, in_order, out_order)
    
    for fusion_mode in ['enabled', 'disabled']:
        with nvtx.annotate("bench_fusion.Fusion({fusion_mode})".format(fusion_mode=fusion_mode), color="purple"):
            fusion=bench_fusion.Fusion()
            fusion.setup(fusion_mode)
            with nvtx.annotate("bench_fusion.Fusion.add_10_times({fusion_mode})".format(fusion_mode=fusion_mode), color="green"):
                fusion.time_fusion_add_10_times(fusion_mode)
            with nvtx.annotate("bench_fusion.Fusion.fusion_loops({fusion_mode})".format(fusion_mode=fusion_mode), color="green"):
                fusion.time_fusion_loops(fusion_mode)
            with nvtx.annotate("bench_fusion.Fusion.fusion_lstm_grad_grad({fusion_mode})".format(fusion_mode=fusion_mode), color="green"):
                fusion.time_fusion_lstm_grad_grad(fusion_mode)

