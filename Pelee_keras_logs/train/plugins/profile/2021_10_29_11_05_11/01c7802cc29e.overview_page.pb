?	?5?!g?@?5?!g?@!?5?!g?@	?f$WvA@?f$WvA@!?f$WvA@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?5?!g?@?^F?????1????q(?@IJV??1?@YYj??pX?@r0*	@5^? @A2O
Iterator::Root::Prefetch?bG?Pb?@!{?	??X@)?bG?Pb?@1{?	??X@:Preprocessing2E
Iterator::Root*?=%gb?@!      Y@)F[?D?A??1<????`?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 34.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?f$WvA@I4?2?a"@Q???c??K@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^F??????^F?????!?^F?????      ??!       "	????q(?@????q(?@!????q(?@*      ??!       2      ??!       :	JV??1?@JV??1?@!JV??1?@B      ??!       J	Yj??pX?@Yj??pX?@!Yj??pX?@R      ??!       Z	Yj??pX?@Yj??pX?@!Yj??pX?@b      ??!       JGPUY?f$WvA@b q4?2?a"@y???c??K@?"H
*model/bbn_features_transition2_conv/Conv2DConv2D3????<??!3????<??0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputZ???????!F??`??0"H
*model/bbn_features_transition1_conv/Conv2DConv2D?2?  ???!ߗ?N????0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput&???g??!??N????0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?Ǒ٤??!)?=??~??0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??#?}??!?@?!???0"H
*model/bbn_features_transition3_conv/Conv2DConv2D?'?+??!??c??y??0"M
/model/bbn_features_stemblock_stem2a_conv/Conv2DConv2D?^?^/??!??M????0"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput>??????!????<???0"s
Hgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInputeh?dމ?!??t?/j??0Q      Y@Yժ?????a?B ??X@q???═?y?d$?"?

host?Your program is HIGHLY input-bound because 34.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?9.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 