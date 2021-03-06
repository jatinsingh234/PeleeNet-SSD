?	`[?????@`[?????@!`[?????@	?Ox?hU@?Ox?hU@!?Ox?hU@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0`[?????@Tt$??`@1,?F<وr@I?V??(D@Y??>???@r0*	????w?A2O
Iterator::Root::Prefetch ??ƅ?@!?:????X@) ??ƅ?@1?:????X@:Preprocessing2E
Iterator::Root;ǀ???@!      Y@)?W:?%??1?TV?Y?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 85.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9?Ox?hU@I ?N? @Q?ک?v?(@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Tt$??`@Tt$??`@!Tt$??`@      ??!       "	,?F<وr@,?F<وr@!,?F<وr@*      ??!       2      ??!       :	?V??(D@?V??(D@!?V??(D@B      ??!       J	??>???@??>???@!??>???@R      ??!       Z	??>???@??>???@!??>???@b      ??!       JGPUY?Ox?hU@b q ?N? @y?ک?v?(@?"|
Qgradient_tape/model/bbn_features_stemblock_stem2b_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputp??AO???!p??AO???0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput??s?r??!xJ?Zn???0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter"?ˁӫ??!ģ?,i??0"H
*model/bbn_features_transition1_conv/Conv2DConv2DsҖ?,E??!aX?6w???0"H
*model/bbn_features_transition2_conv/Conv2DConv2D-??*r҃?!l?/????0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput????s???!Q??Z4??0"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?m~?c??!?[?j????0"H
*model/bbn_features_transition3_conv/Conv2DConv2D?8??ށ?!?D?`???0"p
Fgradient_tape/model/bbn_features_transition1_norm/FusedBatchNormGradV3FusedBatchNormGradV3?.?F???!?h!?ս??"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???????!???0Q      Y@YD??<[???a?äB?X@qn>}??b??yq?w??B]?"?	
host?Your program is HIGHLY input-bound because 85.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nono*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 