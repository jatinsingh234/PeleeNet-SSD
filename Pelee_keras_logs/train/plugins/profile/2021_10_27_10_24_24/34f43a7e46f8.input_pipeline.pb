	??'Hl?o@??'Hl?o@!??'Hl?o@      ??!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??'Hl?o@?N?Z??@1B
?B.?j@A{??x?I?0?:9=@@r0*	\???(?\@2E
Iterator::RootuYLl>???!      Y@)???!6X??1???ĸ?K@:Preprocessing2O
Iterator::Root::PrefetchԹ?????!x78;GwF@)Թ?????1x78;GwF@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.moderate"?12.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI ?%G?-@Q?A?EU@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?N?Z??@?N?Z??@!?N?Z??@      ??!       "	B
?B.?j@B
?B.?j@!B
?B.?j@*      ??!       2	{??x?{??x?!{??x?:	?0?:9=@@?0?:9=@@!?0?:9=@@B      ??!       J      ??!       R      ??!       Z      ??!       b      ??!       JGPUb q ?%G?-@y?A?EU@