?	K???l?@K???l?@!K???l?@	???0ͯ>@???0ͯ>@!???0ͯ>@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0K???l?@???p?v??1
??r??@I?a?Q??@Y??ޫ&ؗ@r0*	l??? e7A2O
Iterator::Root::Prefetchyt#,???@!{`???X@)yt#,???@1{`???X@:Preprocessing2E
Iterator::Root???????@!      Y@)~5????1?[z??	k?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 30.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.moderate"?10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9???0ͯ>@I??Z"YQ$@Q?ÓM@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	???p?v?????p?v??!???p?v??      ??!       "	
??r??@
??r??@!
??r??@*      ??!       2      ??!       :	?a?Q??@?a?Q??@!?a?Q??@B      ??!       J	??ޫ&ؗ@??ޫ&ؗ@!??ޫ&ؗ@R      ??!       Z	??ޫ&ؗ@??ޫ&ؗ@!??ޫ&ؗ@b      ??!       JGPUY???0ͯ>@b q??Z"YQ$@y?ÓM@?"H
*model/bbn_features_transition1_conv/Conv2DConv2D?;V???!?;V???0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter???X??!?to???0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputo?gx???!??????0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?I?-2	??!??????0"H
*model/bbn_features_transition3_conv/Conv2DConv2D???????!??>a????0"H
*model/bbn_features_transition2_conv/Conv2DConv2D?_??Tr??!ܳ??6??0"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputW???ǻ??!'?7??[??0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputey2?4??!T???O???0"s
Hgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput??8P???!`?CT~??0"y
Mgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter{??O?u??!?C(?%??0Q      Y@Y?G_D?M??a\?].Y?X@q?CE?͙?y?4???\%?"?

host?Your program is HIGHLY input-bound because 30.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
moderate?10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 