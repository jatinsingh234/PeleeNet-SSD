?	?rf?m@?rf?m@!?rf?m@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?rf?m@??]/M??1??nf??O@I????D?e@r0*	X9??vD{@2E
Iterator::Root?;?????!      Y@)X???!??1?-?ц?V@:Preprocessing2O
Iterator::Root::Prefetch??̰Q??!}?Vs??#@)??̰Q??1}?Vs??#@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?73.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIXK3~QR@Q??2??:@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??]/M????]/M??!??]/M??      ??!       "	??nf??O@??nf??O@!??nf??O@*      ??!       2      ??!       :	????D?e@????D?e@!????D?e@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb qXK3~QR@y??2??:@?"6
model/re_lu_113/Relu_FusedConv2D?Xi?*?}?!?Xi?*?}?"|
Qgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput??L??{?!?3.$?̌?0"H
*model/bbn_features_transition3_conv/Conv2DConv2D?E?~?z?!C?B]&??0"H
*model/bbn_features_transition1_conv/Conv2DConv2D??bq?ry?!5js????0"t
Jgradient_tape/model/bbn_features_stemblock_stem1_norm/FusedBatchNormGradV3FusedBatchNormGradV3?m?Q?x?!?b=??Ҡ?"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?$???Tw?!p?W?J???0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?????v?!r?.?~???0"^
8model/bbn_features_stemblock_stem1_norm/FusedBatchNormV3FusedBatchNormV3?Ywj?,v?!??}*_??"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?!?-:?u?!??7p??0"_
9model/bbn_features_stemblock_stem2a_norm/FusedBatchNormV3FusedBatchNormV3?????yu?!???:???Q      Y@Y?<?????a??.?X@qwK-???E@yƝK????"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?73.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?43.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 