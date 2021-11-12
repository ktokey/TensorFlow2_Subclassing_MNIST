import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb

#============================================================
# tensorflow2.xでのGPUの設定
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    #
    for k in range(len(physical_devices)):
        tf.config.set_visible_devices(physical_devices[k], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
else:
    print("Not enough GPU hardware devices available")
#============================================================

#----------------------------
# parameter setting
visual_path = 'visualization'
checkpoint_path = 'checkpoint'
checkpoint_file = 'weights_nn.ckpt'

BATCH_SIZE = 128
Epochs = 10

isVisualize = True
isLoadModel = False
isTraining = True

# ディレクトリが存在しない場合は作成
if not os.path.exists(visual_path):
    os.makedirs(visual_path)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
#----------------------------

#----------------------------
# データの作成

# MNISTデータの読み込み
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 画像の正規化
x_train = x_train / 255.
x_test = x_test / 255.

# チャネル軸を追加
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# データ供給ライブラリを利用
TRAIN_SIZE = int(0.8 * len(x_train))
TRAIN_DATA = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(x_train.shape[0])
train_data = TRAIN_DATA.take(TRAIN_SIZE).batch(BATCH_SIZE)
val_data = TRAIN_DATA.skip(TRAIN_SIZE).batch(BATCH_SIZE)
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(x_test.shape[0]).batch(BATCH_SIZE)

if isVisualize:
    # 画像を可視化
    fig = plt.figure()
    for i in range(20):
        fig.add_subplot(4,5,i+1)
        plt.imshow(x_train[i,:,:,0], vmin=0, vmax=1, cmap='gray')
        plt.title(f'number:{y_train[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{visual_path}/trainImgs.pdf')

    fig = plt.figure()
    for i in range(20):
        fig.add_subplot(4,5,i+1)
        plt.imshow(x_test[i,:,:,0], vmin=0, vmax=1, cmap='gray')
        plt.title(f'number:{y_test[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{visual_path}/testImgs.pdf')
#----------------------------

#----------------------------
# Subclassingを用いたネットワークの定義

class myModel(tf.keras.Model):
    def __init__(self):
        super(myModel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation='sigmoid')
        self.dense2 = tf.keras.layers.Dense(50, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            # 予測
            y_pred = self(x, training=True)
            # train using gradients 
            trainable_vars = self.trainable_variables
            # loss
            loss = self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        # 勾配を用いた学習
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, trainable_vars) if grad is not None)
        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y_true = data
        # 予測
        y_pred = self(x, training=False)
        # loss
        self.compiled_loss(y_true, y_pred, regularization_losses=self.losses)
        # update metrics
        self.compiled_metrics.update_state(y_true, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def predict_step(self, x):
        # 予測
        y_pred = self(x, training=False)
        return y_pred
#----------------------------

#----------------------------
# モデルの設定
model = myModel()

# 学習方法の設定
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'], run_eagerly=True)

if isLoadModel:
    try:
        # load trained parameters
        model.load_weights(f'{checkpoint_path}/{checkpoint_file}')
    except tf.errors.NotFoundError:
        print('Could not load weights!')
    else:
        print('load weights')

if isTraining:
    # make checkpoint callback to save trained parameters
    callback = tf.keras.callbacks.ModelCheckpoint(f'{checkpoint_path}/{checkpoint_file}', monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='auto', min_delta=0, verbose=1)
    # 学習
    history = model.fit(train_data, validation_data=val_data, epochs=Epochs, callbacks=[callback, early_stopping])

if isVisualize and isTraining:
    # 学習曲線をプロット
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure()
    plt.plot(np.arange(Epochs), loss, 'bo-', label='training loss')
    plt.plot(np.arange(Epochs), val_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.savefig(f'{visual_path}/loss.pdf')

    # 正解率をプロット
    accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']

    plt.figure()
    plt.plot(np.arange(Epochs), accuracy, 'bo-', label='training accuracy')
    plt.plot(np.arange(Epochs), val_accuracy, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.savefig(f'{visual_path}/accuracy.pdf')

# 検証
result = model.evaluate(test_data)
print(dict(zip(model.metrics_names, result)))

# 予測（今回はx_testを入力していますが、コンペ等では提出用データを予測します）
pred_data = tf.data.Dataset.from_tensor_slices(x_test).batch(BATCH_SIZE)
pred = model.predict(pred_data).argmax(axis=1)