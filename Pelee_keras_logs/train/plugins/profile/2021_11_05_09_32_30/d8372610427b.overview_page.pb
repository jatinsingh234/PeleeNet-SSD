?	???Q?@???Q?@!???Q?@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???Q?@1}?!8.??1X??k?@A????Ŋ??I?RIh?@r0*	?G? GA2f
/Iterator::Root::Prefetch::FlatMap[0]::Generatoran??@!??Y??X@)an??@1??Y??X@:Preprocessing2O
Iterator::Root::Prefetch稣?j??!????T)^?)稣?j??1????T)^?:Preprocessing2E
Iterator::Root????·?!??I8i?)jin????1"?˹?FT?:Preprocessing2X
!Iterator::Root::Prefetch::FlatMap/?Hō?@!`lُ??X@)?:pΈr?1?????#?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI0C???%)@Q?WF%C?U@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	1}?!8.??1}?!8.??!1}?!8.??      ??!       "	X??k?@X??k?@!X??k?@*      ??!       2	????Ŋ??????Ŋ??!????Ŋ??:	?RIh?@?RIh?@!?RIh?@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q0C???%)@y?WF%C?U@?"H
*model/bbn_features_transition1_conv/Conv2DConv2D??@?d??!??@?d??0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput'4ܒX???!?]W/&??0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilterUJ????!0???a??0"H
*model/bbn_features_transition3_conv/Conv2DConv2Dv??Ǖ??!?ڔ?ǽ?0"H
*model/bbn_features_transition2_conv/Conv2DConv2Du?]?S??!?#vb1??0"L
.model/bbn_features_stemblock_stem1_conv/Conv2DConv2D
?'??V??!?????0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput??ϰ)??!?+??=???0"|
Qgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput=F2ys??!r?Fଂ??0"M
/model/bbn_features_stemblock_stem2a_conv/Conv2DConv2D???`3??!?KJ?	??0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???.P??!%?)?S??0Q      Y@YwAM\l???a_???>?X@q?xz???y\Y?@s? ?"?

device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?12.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 