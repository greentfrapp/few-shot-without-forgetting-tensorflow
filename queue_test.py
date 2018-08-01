import tensorflow as tf

all_filenames = ['a', 'b', 'c']
list2 = ['1', '2', '3']

q1 = tf.train.string_input_producer(tf.convert_to_tensor(all_filenames), shuffle=True, seed=42).dequeue()

q2 = tf.train.string_input_producer(tf.convert_to_tensor(list2), shuffle=True, seed=42).dequeue()

images, labels = tf.train.batch(
	[q1,q2],
	batch_size = 2,
	num_threads=1,
	capacity=6,
	)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for i in range(10):
	i, l = sess.run([images, labels])
	print(i)
	print(l)