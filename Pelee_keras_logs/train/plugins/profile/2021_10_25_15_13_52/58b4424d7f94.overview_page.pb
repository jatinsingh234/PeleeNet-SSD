?	y!?>?@y!?>?@!y!?>?@	?C?=hG@?C?=hG@!?C?=hG@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0y!?>?@???????1?A?"L?@IR'??p?~@Y0????@r0*	㥛?THA2O
Iterator::Root::Prefetch????Ԛ?@!'W???X@)????Ԛ?@1'W???X@:Preprocessing2E
Iterator::Root???????@!      Y@)Xr?߰?1>?lp?$a?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 46.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?C?=hG@I8ܞF@Q???H ?F@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????????!???????      ??!       "	?A?"L?@?A?"L?@!?A?"L?@*      ??!       2      ??!       :	R'??p?~@R'??p?~@!R'??p?~@B      ??!       J	0????@0????@!0????@R      ??!       Z	0????@0????@!0????@b      ??!       JGPUY?C?=hG@b q8ܞF@y???H ?F@?"H
*model/bbn_features_transition1_conv/Conv2DConv2D8??B?^??!8??B?^??0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter
????T??!!?4??ٷ?0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput)kM?2??!?a?pHf??0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??jq??!?u?????0"H
*model/bbn_features_transition2_conv/Conv2DConv2D??ezܓ?!??a?s0??0"H
*model/bbn_features_transition3_conv/Conv2DConv2D?R:????!UI1????0"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?r~??Ց?!???n????0"L
.model/bbn_features_stemblock_stem1_conv/Conv2DConv2D{ѹ?k???!???.??0"L
.model/bbn_features_stemblock_stem3_conv/Conv2DConv2D̠??}??!?2`??3??0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput??)'C=??!?iEz[??0Q      Y@Y?G_D?M??a\?].Y?X@q???Ǒv??yA?h=??#?"?

host?Your program is HIGHLY input-bound because 46.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?7.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 