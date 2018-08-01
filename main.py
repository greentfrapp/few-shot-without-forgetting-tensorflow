from __future__ import print_function
try:
	raw_input
except:
	raw_input = input

from time import strftime
import os
import json
from comet_ml import Experiment
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from absl import flags
from absl import app

from models import CNN_miniimagenet
from data_generator import DataGenerator


FLAGS = flags.FLAGS

# Commands
flags.DEFINE_bool('pretrain', False, 'Train')
flags.DEFINE_bool('train', False, 'Train')
flags.DEFINE_bool('test', False, 'Test')

# Task parameters
flags.DEFINE_string('datasource', 'omniglot', 'omniglot or sinusoid (miniimagenet WIP)')
flags.DEFINE_integer('num_classes', 5, 'Number of classes per task eg. 5-way refers to 5 classes')
flags.DEFINE_integer('num_shot_train', None, 'Number of training samples per class per task eg. 1-shot refers to 1 training sample per class')
flags.DEFINE_integer('num_shot_test', None, 'Number of test samples per class per task')

# Training parameters
flags.DEFINE_integer('steps', None, 'Number of metatraining iterations')
flags.DEFINE_integer('meta_batch_size', 32, 'Batchsize for metatraining')
flags.DEFINE_float('meta_lr', 0.001, 'Meta learning rate')
flags.DEFINE_integer('validate_every', 500, 'Frequency for metavalidation and saving')
flags.DEFINE_string('savepath', 'saved_models/', 'Path to save or load models')
flags.DEFINE_string('logdir', 'logs/', 'Path to save Tensorboard summaries')

# Logging parameters
flags.DEFINE_integer('print_every', 100, 'Frequency for printing training loss and accuracy')


def main(unused_args):

	if FLAGS.pretrain:
		data_generator = DataGenerator(
			datasource='miniimagenet',
			num_classes=64,
			num_samples_per_class=1,
			batch_size=8,
			test_set=False,
			mode='pretrain',
		)
		image_tensor, label_tensor = data_generator.make_data_tensor(train=True)
		input_tensors = {
			'inputs': image_tensor, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'labels': label_tensor, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
		}
		model = CNN_miniimagenet('model', num_classes=64, input_tensors=input_tensors, mode='pretrain')
		
		# Validation
		data_generator = DataGenerator(
			datasource='miniimagenet',
			num_classes=5,
			num_samples_per_class=2,
			batch_size=32,
			test_set=False,
			mode='fewshot',
		)
		metaval_image_tensor, metaval_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(metaval_image_tensor, [0, 0, 0], [-1, 5, -1])
		test_inputs = tf.slice(metaval_image_tensor, [0, 5, 0], [-1, -1, -1])
		train_labels = tf.slice(metaval_label_tensor, [0, 0, 0], [-1, 5, -1])
		test_labels = tf.slice(metaval_label_tensor, [0, 5, 0], [-1, -1, -1])
		metaval_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}
		val_model = CNN_miniimagenet('model', num_classes=5, input_tensors=metaval_input_tensors, mode='fewshot')

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		min_val_loss = np.inf
		steps = 200000
		LUT_lr = [(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)]
		try:
			for step in np.arange(steps):
				for i, lookup in enumerate(LUT_lr):
					if step < lookup[0]:
						break
				lr = LUT_lr[i-1][1]
				loss, accuracy, _ = sess.run([model.loss, model.accuracy, model.optimize], {model.lr: lr})
				if step > 0 and step % FLAGS.print_every == 0:
					print('Step #{} - Loss : {:.3f} - Acc : {:.3f}'.format(step, loss, accuracy))
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (steps - 1)):
					val_loss, val_accuracy = sess.run([val_model.loss, val_model.test_accuracy])
					print('Validation Step - Loss : {:.3f} - Acc : {:.3f}'.format(val_loss, val_accuracy))
					if step == (steps - 1):
						print('Training complete!')
					if val_loss < min_val_loss:
						model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
						min_val_loss = val_loss
						
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input('\nSave latest model at Step #{}? (y/n)\n'.format(step))
			if response == 'y':
				model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print('Latest model not saved.')

	elif FLAGS.train:

		data_generator = DataGenerator(
			datasource='miniimagenet',
			num_classes=5,
			num_samples_per_class=2,
			batch_size=4,
			test_set=False,
		)
		metatrain_image_tensor, metatrain_label_tensor = data_generator.make_data_tensor(train=True)
		train_inputs = tf.slice(metatrain_image_tensor, [0, 0, 0], [-1, 5, -1])
		test_inputs = tf.slice(metatrain_image_tensor, [0, 5, 0], [-1, -1, -1])
		train_labels = tf.slice(metatrain_label_tensor, [0, 0, 0], [-1, 5, -1])
		test_labels = tf.slice(metatrain_label_tensor, [0, 5, 0], [-1, -1, -1])
		metatrain_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}
		model = CNN_miniimagenet('model', num_classes=5, input_tensors=metatrain_input_tensors, mode='fewshot', use_attention=True)
		
		# Validation
		data_generator = DataGenerator(
			datasource='miniimagenet',
			num_classes=5,
			num_samples_per_class=2,
			batch_size=32,
			test_set=False,
			mode='fewshot',
		)
		metaval_image_tensor, metaval_label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(metaval_image_tensor, [0, 0, 0], [-1, 5, -1])
		test_inputs = tf.slice(metaval_image_tensor, [0, 5, 0], [-1, -1, -1])
		train_labels = tf.slice(metaval_label_tensor, [0, 0, 0], [-1, 5, -1])
		test_labels = tf.slice(metaval_label_tensor, [0, 5, 0], [-1, -1, -1])
		metaval_input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}
		val_model = CNN_miniimagenet('model', num_classes=5, input_tensors=metaval_input_tensors, mode='fewshot', use_attention=True)

		sess = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		tf.train.start_queue_runners()

		min_val_loss = np.inf
		steps = 200000
		LUT_lr = [(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)]
		try:
			for step in np.arange(steps):
				for i, lookup in enumerate(LUT_lr):
					if step < lookup[0]:
						break
				lr = LUT_lr[i-1][1]
				loss, accuracy, _ = sess.run([model.loss, model.test_accuracy, model.optimize])
				if step > 0 and step % FLAGS.print_every == 0:
					print('Step #{} - Loss : {:.3f} - Acc : {:.3f}'.format(step, loss, accuracy))
				if step > 0 and (step % FLAGS.validate_every == 0 or step == (steps - 1)):
					val_loss, val_accuracy = sess.run([val_model.loss, val_model.test_accuracy])
					print('Validation Step - Loss : {:.3f} - Acc : {:.3f}'.format(val_loss, val_accuracy))
					if step == (steps - 1):
						print('Training complete!')
					if val_loss < min_val_loss:
						model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
						min_val_loss = val_loss
						
		# Catch Ctrl-C event and allow save option
		except KeyboardInterrupt:
			response = raw_input('\nSave latest model at Step #{}? (y/n)\n'.format(step))
			if response == 'y':
				model.save(sess, FLAGS.savepath, global_step=step, verbose=True)
			else:
				print('Latest model not saved.')

	elif FLAGS.test:

		NUM_TEST_SAMPLES = 2000

		num_shot_train = FLAGS.num_shot_train or 1
		num_shot_test = FLAGS.num_shot_test or 1
		num_classes_test = 5

		data_generator = DataGenerator(
			datasource='miniimagenet',
			num_classes=num_classes_test,
			num_samples_per_class=num_shot_train+num_shot_test,
			batch_size=1,
			test_set=True,
		)

		image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
		train_inputs = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes_test*num_shot_train, -1])
		test_inputs = tf.slice(image_tensor, [0, num_classes_test*num_shot_train, 0], [-1, -1, -1])
		train_labels = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes_test*num_shot_train, -1])
		test_labels = tf.slice(label_tensor, [0, num_classes_test*num_shot_train, 0], [-1, -1, -1])
		input_tensors = {
			'train_inputs': train_inputs, # batch_size, num_classes * (num_samples_per_class - update_batch_size), 28 * 28
			'train_labels': train_labels, # batch_size, num_classes * (num_samples_per_class - update_batch_size), num_classes
			'test_inputs': test_inputs, # batch_size, num_classes * update_batch_size, 28 * 28
			'test_labels': test_labels, # batch_size, num_classes * update_batch_size, num_classes
		}

		model = CNN_miniimagenet('model', num_classes=num_classes_test, input_tensors=input_tensors, mode='fewshot', use_attention=True)

		sess = tf.InteractiveSession()
		model.load(sess, FLAGS.savepath, verbose=True)
		tf.train.start_queue_runners()
		
		accuracy_list = []

		for task in np.arange(NUM_TEST_SAMPLES):
			accuracy = sess.run(model.test_accuracy)
			accuracy_list.append(accuracy)
			if task > 0 and task % 100 == 0:
				print('Metatested on {} tasks...'.format(task))

		avg = np.mean(accuracy_list)
		stdev = np.std(accuracy_list)
		ci95 = 1.96 * stdev / np.sqrt(NUM_TEST_SAMPLES)

		print('\nEnd of Test!')
		print('Accuracy                : {:.4f}'.format(avg))
		print('StdDev                  : {:.4f}'.format(stdev))
		print('95% Confidence Interval : {:.4f}'.format(ci95))


if __name__ == '__main__':
	app.run(main)