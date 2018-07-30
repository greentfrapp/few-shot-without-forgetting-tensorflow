"""
Architecture for Few-Shot miniImagenet
"""

import tensorflow as tf
import numpy as np

from .BaseModel import BaseModel


class FeatureExtractor(object):

	def __init__(self, inputs, is_training=None):
		self.inputs = inputs
		# self.is_training = is_training
		self.n_filters = [64, 64, 128, 128]
		self.dropout_rate = [None, None, 0.1, 0.3]
		with tf.variable_scope("extractor", reuse=tf.AUTO_REUSE):
			self.build_model()

	def build_model(self):
		running_output = self.inputs
		for i, filters in enumerate(self.n_filters):
			conv = tf.layers.conv2d(
				inputs=running_output,
				filters=filters,
				kernel_size=(3, 3),
				strides=(1, 1),
				padding="same",
				activation=None,
				kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
				name="conv_{}".format(i),
				reuse=tf.AUTO_REUSE,
			)
			norm = tf.contrib.layers.batch_norm(
				inputs=conv,
				activation_fn=None,
				reuse=tf.AUTO_REUSE,
				scope="model/extractor/norm_{}".format(i),
				# is_training=self.is_training, # should be True for both metatrain and metatest
			)
			maxpool = tf.layers.max_pooling2d(
				inputs=norm,
				pool_size=(2, 2),
				strides=(2, 2),
				padding="valid",
			)
			if i == len(self.n_filters) - 1:
				relu = maxpool
			else:
				relu = tf.nn.relu(
					features=maxpool,
				)
			# if self.dropout_rate[i] is None:
			# 	dropout = relu
			# else:
			# 	dropout = tf.layers.dropout(
			# 		inputs=relu,
			# 		rate=self.dropout_rate[i],
			# 		training=self.is_training,
			# 	)
			running_output = relu

		running_output = tf.reshape(running_output, [-1, 5*5*128])
		running_output = tf.nn.l2_normalize(running_output, dim=-1)

		self.output = running_output # shape = (meta_batch_size*num_shot_train, 5*5*32)

class CNN_miniimagenet(BaseModel):

	def __init__(self, name, num_classes, input_tensors=None, mode='fewshot'):
		super(CNN_miniimagenet, self).__init__()
		self.name = name
		self.num_classes = num_classes
		self.mode = mode
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			self.build_model(input_tensors)
			variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
			self.saver = tf.train.Saver(var_list=variables, max_to_keep=5)

	def build_model(self, input_tensors=None):

		if self.mode == 'pretrain':

			self.inputs = tf.reshape(input_tensors['inputs'], [-1, 84, 84, 3])
			self.labels = tf.reshape(input_tensors['labels'], [-1, self.num_classes])

			feature_extractor = FeatureExtractor(self.inputs)

			classifier_weights = tf.get_variable(
				name='classifier_weights',
				shape=(self.num_classes, 5*5*128),
				dtype=tf.float32,
			)

			scale = tf.Variable(
				initial_value=10.,
				name='scale',
				dtype=tf.float32,
			)

			classifier_weights = tf.nn.l2_normalize(classifier_weights, dim=-1)

			self.logits = scale * tf.matmul(feature_extractor.output, classifier_weights, transpose_b=True)

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
			self.optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
			self.accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.labels, axis=1), predictions=tf.argmax(self.logits, axis=1))

		elif self.mode == 'fewshot':

			self.train_inputs = tf.reshape(input_tensors['train_inputs'], [-1, 84, 84, 3])
			self.test_inputs = tf.reshape(input_tensors['test_inputs'], [-1, 84, 84, 3])
			self.train_labels = tf.reshape(input_tensors['train_labels'], [-1, self.num_classes])
			self.test_labels = tf.reshape(input_tensors['test_labels'], [-1, self.num_classes])

			batchsize = tf.shape(input_tensors['train_inputs'])[0]
			num_shot_train = tf.shape(input_tensors['train_inputs'])[1]
			num_shot_test = tf.shape(input_tensors['test_inputs'])[1]

			# Extract training features
			train_feature_extractor = FeatureExtractor(self.train_inputs)
			train_labels = tf.reshape(self.train_labels, [batchsize, -1, self.num_classes])
			train_features = tf.reshape(train_feature_extractor.output, [batchsize, -1, 5*5*128])
			self.train_features = train_features
			# Take mean of features for each class
			output_weights = tf.matmul(train_labels, train_features, transpose_a=True) / tf.expand_dims(tf.reduce_sum(train_labels, axis=1), axis=-1)
			
			# Calculate class weights with attention
			# with tf.variable_scope("attention"):
			# 	train_embed = tf.layers.dense(
			# 		inputs=output_weights,
			# 		units=self.hidden,
			# 		activation=None,
			# 		name="train_embed",
			# 	)
			# 	for i in np.arange(self.attention_layers):
			# 		train_embed, _ = self.attention(
			# 			query=train_embed,
			# 			key=train_embed,
			# 			value=train_embed,
			# 		)
			# 		dense = tf.layers.dense(
			# 			inputs=train_embed,
			# 			units=self.hidden * 2,
			# 			activation=tf.nn.relu,
			# 			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			# 			name="attention_layer{}_dense0".format(i),
			# 		)
			# 		train_embed += tf.layers.dense(
			# 			inputs=dense,
			# 			units=self.hidden,
			# 			activation=None,
			# 			kernel_initializer=tf.contrib.layers.xavier_initializer(),
			# 			name="attention_layer{}_dense1".format(i)
			# 		)
			# 		train_embed = tf.contrib.layers.layer_norm(train_embed, begin_norm_axis=2)

			# 	class_weights = tf.layers.dense(
			# 		inputs=train_embed,
			# 		units=5*5*32,
			# 		activation=None,
			# 		kernel_initializer=tf.contrib.layers.xavier_initializer(),
			# 	)

			# Extract test features
			test_feature_extractor = FeatureExtractor(self.test_inputs)
			test_features = tf.reshape(test_feature_extractor.output, [batchsize, -1, 5*5*128])
			
			class_weights = output_weights
			class_weights /= tf.norm(class_weights, axis=-1, keep_dims=True)
			test_features /= tf.norm(test_features, axis=-1, keep_dims=True)

			self.scale = tf.Variable(
				initial_value=10.,
				name="scale",
				dtype=tf.float32,
			)

			logits = tf.matmul(test_features, class_weights, transpose_b=True)
			logits = logits * self.scale
			self.logits = logits = tf.reshape(logits, [-1, self.num_classes])

			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.test_labels, logits=self.logits))
			self.optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)
			self.test_accuracy = tf.contrib.metrics.accuracy(labels=tf.argmax(self.test_labels, axis=1), predictions=tf.argmax(self.logits, axis=1))

		else:
			raise ValueError('Unrecognized mode value for CNN_miniimagenet object')