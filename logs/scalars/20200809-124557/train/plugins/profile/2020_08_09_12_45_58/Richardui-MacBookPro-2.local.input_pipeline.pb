	�Q����?�Q����?!�Q����?	���[��B@���[��B@!���[��B@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�Q����?Zd;�O��?A��"��~�?Y��~j�t�?*	     �x@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���x�&�?!��JG�@@)����Mb�?1h *�3@@:Preprocessing2F
Iterator::ModelH�z�G�?!-n���A@)!�rh���?18a~W��<@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat;�O��n�?!ud�@T:2@)�Q���?1q�w��1@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::MapV-��?!�Z܄�]=@)�I+��?1���JG&@:Preprocessing2U
Iterator::Model::ParallelMapV2�I+��?!���JG@)�I+��?1���JG@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat{�G�z�?!��td�@@){�G�z�?1��td�@@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�~j�t�x?!�0���M�?)�~j�t�x?1�0���M�?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range����Mb`?!h *�3�?)����Mb`?1h *�3�?:Preprocessing2R
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::TensorSlice:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 37.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t11.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9���[��B@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	Zd;�O��?Zd;�O��?!Zd;�O��?      ��!       "      ��!       *      ��!       2	��"��~�?��"��~�?!��"��~�?:      ��!       B      ��!       J	��~j�t�?��~j�t�?!��~j�t�?R      ��!       Z	��~j�t�?��~j�t�?!��~j�t�?JCPU_ONLYY���[��B@b 