import numpy as np
import tensorflow as tf
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Вимкнення eager execution для сумісності зі старим кодом
tf.compat.v1.disable_eager_execution()

# Параметри
n_samples = 1000
batch_size = 100
num_steps = 20000

# Створення даних (x в інтервалі [1, 10], як у методичці)
X_data = np.random.uniform(1, 10, (n_samples, 1))
y_data = 2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))

# Placeholder
X = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, 1))

# Змінні моделі
with tf.compat.v1.variable_scope('linear-regression'):
    k = tf.Variable(tf.compat.v1.random_normal((1, 1)), name='slope')
    b = tf.Variable(tf.zeros((1,)), name='bias')

# Модель та функція втрат
y_pred = tf.matmul(X, k) + b
loss = tf.reduce_sum((y - y_pred) ** 2)

# Оптимізатор (з learning rate)
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.0001).minimize(loss)

display_step = 100

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for i in range(num_steps):
        indices = np.random.choice(n_samples, batch_size)
        X_batch = X_data[indices]
        y_batch = y_data[indices]

        _, loss_val, k_val, b_val = sess.run(
            [optimizer, loss, k, b],
            feed_dict={X: X_batch, y: y_batch}
        )

        if (i + 1) % display_step == 0:
            print(
                f'Епоха {i+1}: {loss_val:.8f}, k={k_val[0][0]:.4f}, b={b_val[0]:.4f}')

print(f"\nІстинні: k=2.0, b=1.0")
print(f"Навчені: k={k_val[0][0]:.4f}, b={b_val[0]:.4f}")
