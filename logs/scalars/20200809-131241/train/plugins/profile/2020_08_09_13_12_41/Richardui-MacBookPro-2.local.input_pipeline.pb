	�Q����?�Q����?!�Q����?	�tk~X�<@�tk~X�<@!�tk~X�<@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�Q����?ˡE����?AX9��v��?YV-��?*	     pv@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�O��n�?!����D@)�n����?10;�%�[C@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatP��n��?!U��aq�A@)P��n��?1U��aq�A@:Preprocessing2F
Iterator::Model�� �rh�?!Ժ�j�2@)���Q��?1C41��0@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!c���'�@)�~j�t��?1� ���
@:Preprocessing2U
Iterator::Model::ParallelMapV2����Mb�?!�d���@)����Mb�?1�d���@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetchy�&1�|?!�o�(�2�?)y�&1�|?1�o�(�2�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::TensorSlice�~j�t�h?!� ����?)�~j�t�h?1� ����?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����MbP?!�d����?)����MbP?1�d����?:Preprocessing2f
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range:Preprocessing2R
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::TensorSlice:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 28.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t10.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�tk~X�<@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ˡE����?ˡE����?!ˡE����?      ��!       "      ��!       *      ��!       2	X9��v��?X9��v��?!X9��v��?:      ��!       B      ��!       J	V-��?V-��?!V-��?R      ��!       Z	V-��?V-��?!V-��?JCPU_ONLYY�tk~X�<@b 