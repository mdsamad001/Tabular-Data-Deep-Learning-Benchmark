# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experiment to train and evaluate the TabNet model on Forest Covertype."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
from tabnet import data_helper_covertype
import numpy as np
from tabnet import tabnet_model
import tensorflow as tf

from tqdm import tqdm

# Run Tensorflow on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # use CPU

# Training parameters
TRAIN_FILE = "data/train_covertype.csv" #not used
VAL_FILE = "data/val_covertype.csv" #not used
TEST_FILE = "data/test_covertype.csv" #not used
MAX_STEPS = 1000#000000 # set this after import
DISPLAY_STEP = 5#5000
VAL_STEP = 1#000
SAVE_STEP = 40#000
INIT_LEARNING_RATE = 0.02
DECAY_EVERY = 500
DECAY_RATE = 0.95
BATCH_SIZE = 16384 # set this after import
SPARSITY_LOSS_WEIGHT = 0.0001
GRADIENT_THRESH = 2000.0
SEED = 1
PATIENCE = 9999
MAX_EPOCH = 1000 # only used for repeating data for batchloader


def main(train_file, val_file, test_file, 
         train_size, val_size, test_size,
         num_features, num_classes,
         columns, feature_columns, input_fn,
         model_name='tabnet_dataset_test-x', pbar = False
        ):

    # Fix random seeds
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    
    virtual_batch_size = 512
    virtual_batch_size = BATCH_SIZE

    # Define the TabNet model
    tabnet_forest_covertype = tabnet_model.TabNet(
        columns=columns,
        num_features=num_features,
        feature_dim=128,
        output_dim=64,
        num_decision_steps=6,
        relaxation_factor=1.5,
        batch_momentum=0.7,
        virtual_batch_size=virtual_batch_size,
        num_classes=num_classes)

    column_names = sorted(feature_columns)
    # print(
    #     "Ordered column names, corresponding to the indexing in Tensorboard visualization"
    # )
    # for i, col in enumerate(column_names):
    #     print(f"{i} : {col}")

    # Input sampling
    # getting outOfRangeError when setting the num_epoch (no of repeats)
    train_batch = input_fn(
        train_file, num_epochs=-1, shuffle=True, 
        batch_size=BATCH_SIZE, n_buffer=train_size)
    val_batch = input_fn(
        val_file,
        num_epochs=-1,
        shuffle=False,
        batch_size=val_size)
    test_batch = input_fn(
        test_file,
        num_epochs=-1,
        shuffle=False,
        batch_size=test_size)

    train_iter = train_batch.make_initializable_iterator()
    val_iter = val_batch.make_initializable_iterator()
    test_iter = test_batch.make_initializable_iterator()

    feature_train_batch, label_train_batch = train_iter.get_next()
    feature_val_batch, label_val_batch = val_iter.get_next()
    feature_test_batch, label_test_batch = test_iter.get_next()
    
    # tf.print(feature_train_batch)

    # Define the model and losses

    encoded_train_batch, total_entropy = tabnet_forest_covertype.encoder(
        feature_train_batch, reuse=False, is_training=True)

    logits_orig_batch, _ = tabnet_forest_covertype.classify(
        encoded_train_batch, reuse=False)
    
    # print_logits = tf.Print( logits_orig_batch, [logits_orig_batch] , message='logits', )
    
    softmax_orig_key_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_orig_batch, labels=label_train_batch))
    
    # softmax_orig_key_op = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         logits=print_logits, labels=label_train_batch))
    
    
    # print_softmax = tf.Print( softmax_orig_key_op, [softmax_orig_key_op] , message = 'softmax' )

    train_loss_op = softmax_orig_key_op + SPARSITY_LOSS_WEIGHT * total_entropy
    # train_loss_op = print_softmax + SPARSITY_LOSS_WEIGHT * total_entropy
    tf.summary.scalar("Total loss", train_loss_op)

    # Optimization step
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        INIT_LEARNING_RATE,
        global_step=global_step,
        decay_steps=DECAY_EVERY,
        decay_rate=DECAY_RATE)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        gvs = optimizer.compute_gradients(train_loss_op)
        capped_gvs = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
                                        GRADIENT_THRESH), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    # Model evaluation

    # Validation performance
    encoded_val_batch, _ = tabnet_forest_covertype.encoder(
        feature_val_batch, reuse=True, is_training=False)

    _, prediction_val = tabnet_forest_covertype.classify(
        encoded_val_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)
    val_eq_op = tf.equal(predicted_labels, label_val_batch)
    val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
    tf.summary.scalar("Val accuracy", val_acc_op)

    # Test performance
    encoded_test_batch, _ = tabnet_forest_covertype.encoder(
        feature_test_batch, reuse=True, is_training=False)

    _, prediction_test = tabnet_forest_covertype.classify(
        encoded_test_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
    test_eq_op = tf.equal(predicted_labels, label_test_batch)
    test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))
    tf.summary.scalar("Test accuracy", test_acc_op)

    # Training setup
    # model_name = "tabnet_forest_covertype_model"
    init = tf.initialize_all_variables()
    init_local = tf.local_variables_initializer()
    init_table = tf.tables_initializer(name="Initialize_all_tables")
    saver = tf.train.Saver()
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("./tflog/" + model_name, sess.graph)

        sess.run(init)
        sess.run(init_local)
        sess.run(init_table)
        sess.run(train_iter.initializer)
        sess.run(val_iter.initializer)
        sess.run(test_iter.initializer)
        
        # pbar = tqdm(range(1, MAX_STEPS + 1))
        pbar.set_description('starting...')
        # pbar = range(1, MAX_STEPS + 1)
        
        val_acc = test_acc = 0
        best_val = 0.000
        best_session = -1
        patience = PATIENCE if PATIENCE > 0 else 9999
        MIN_STEPS = 100
        tries = MIN_STEPS + patience # run at least up to mean epoch
        
        for step in range(1, MAX_STEPS + 1):            
            _, train_loss, merged_summary = sess.run([train_op, train_loss_op, summaries])
            summary_writer.add_summary(merged_summary, step)

            feed_arr = [
                vars()["summaries"],
                vars()["val_acc_op"],
                # vars()["test_acc_op"]
            ]

            val_arr = sess.run(feed_arr)
            merged_summary = val_arr[0]
            val_acc = val_arr[1]
            # test_acc = val_arr[2]
            
            # add early stopping
            if val_acc >= best_val: # we want models that trained for higher epochs yet have equal/greater valid score
                best_val = val_acc
                best_session = sess
                tries = patience if step > MIN_STEPS else tries # dont replenish tries if it did not train up to min_epochs
            else:
                tries -= 1

            pbar and pbar.set_description(f"Step :{step}/{MAX_STEPS}, Training Loss = {train_loss:.4f}, Val Accuracy: {val_acc:.4f} (best: {best_val:.4f})")
            summary_writer.add_summary(merged_summary, step)

            # if step % SAVE_STEP == 0:
            #     # print('save ignored')
            #     saver.save(sess, "./checkpoints/" + model_name + ".ckpt")
            
            if PATIENCE > 0 and (tries<=0 or best_val == 1):
                # print('No improvement, early stopping at epoch:', epoch)
                break
        
        # print(f'{model_name} finished')
        sess = best_session
        
        predictions = sess.run(predicted_labels)
        test_acc = sess.run(test_acc_op)
        
        return best_val, test_acc, predictions


if __name__ == "__main__":
    app.run(main)
