?	?*l???@?*l???@!?*l???@      ??!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'?*l???@E??S???1??<?@I?V$&???@r0*	=?O??&Z@2E
Iterator::Root?@ CǺ?!      Y@)'???6??1069}?gI@:Preprocessing2O
Iterator::Root::PrefetchX????W??!??Ƃ?H@)X????W??1??Ƃ?H@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?19.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??????3@QP?^?
T@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	E??S???E??S???!E??S???      ??!       "	??<?@??<?@!??<?@*      ??!       2      ??!       :	?V$&???@?V$&???@!?V$&???@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q??????3@yP?^?
T@?"H
*model/bbn_features_transition1_conv/Conv2DConv2Deӥ?b\??!eӥ?b\??0"y
Mgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??","??!A<?qG???0"w
Lgradient_tape/model/bbn_features_transition1_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInput?zYu????!??r???0"y
Mgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter??????!F??J/???0"H
*model/bbn_features_transition3_conv/Conv2DConv2DelP-B??!?񋐗y??0"H
*model/bbn_features_transition2_conv/Conv2DConv2D_p????!??9?????0"w
Lgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputl?Y|m???!L(Ş?M??0"w
Lgradient_tape/model/bbn_features_transition2_conv/Conv2D/Conv2DBackpropInputConv2DBackpropInputD@???O??!T??????0"s
Hgradient_tape/model/ssd_feature_extend1_conv2/Conv2D/Conv2DBackpropInputConv2DBackpropInput?{?6???!?`%q??0"y
Mgradient_tape/model/bbn_features_transition3_conv/Conv2D/Conv2DBackpropFilterConv2DBackpropFilter?f+I??!å??0Q      Y@Y?G_D?M??a\?].Y?X@q?!?=?sR@y%G?.I?$?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?19.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?73.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Kepler)(: B 