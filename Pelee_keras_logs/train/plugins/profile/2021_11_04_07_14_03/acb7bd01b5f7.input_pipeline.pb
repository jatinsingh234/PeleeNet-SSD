	?Fˁ?*?@?Fˁ?*?@!?Fˁ?*?@	
?)[?V@
?)[?V@!
?)[?V@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9?Fˁ?*?@?_???!@1Wya1o@A???n??I??a?7?)@Y???_I?@r0*	??zg?FA2O
Iterator::Root::Prefetch??sb??@!3?????X@)??sb??@13?????X@:Preprocessing2E
Iterator::Rooti?????@!      Y@)Z??8???1M??J\:U?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 91.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9
?)[?V@I8#i)????Q??3?H?@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?_???!@?_???!@!?_???!@      ??!       "	Wya1o@Wya1o@!Wya1o@*      ??!       2	???n?????n??!???n??:	??a?7?)@??a?7?)@!??a?7?)@B      ??!       J	???_I?@???_I?@!???_I?@R      ??!       Z	???_I?@???_I?@!???_I?@b      ??!       JGPUY
?)[?V@b q8#i)????y??3?H?@