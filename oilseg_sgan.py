from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import cv2
import math
import time
import json
import random
import argparse
import collections
import numpy as np
import tensorflow as tf
from utils import append_index, load_examples, preprocess, deprocess, save_images, augment, create_generator_Unet
from utils import CROP_SIZE, conv, lrelu, batchnorm
from utils import discrim_loss_sgan, gen_loss_sgan
import time
from compare_results import calculate_accuracy, region_fitting_error
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='Simulation/train', help="path to folder containing images")
parser.add_argument("--mode", default='train', choices=["train", "test", "export"])
parser.add_argument("--output_dir", default='test', help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0,
                    help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=50, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--test_freq", type=int, default=0, help="test model every test_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true",
                    help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--f_type", type=str, default='helinger', help="[f_type choices: [alt, rkl, kl, alpha] ]")
a = parser.parse_args()

EPS = 1e-12
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
                                        "gen_loss, gen_loss_GAN, gen_grads_and_vars, train")


def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2 ** (i + 1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)  # BN
                rectified = lrelu(normalized, 0.2)
                # rectified  = lrelu(convolved, 0.2)#no BN
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = conv(rectified, out_channels=1, stride=1)
            #output = tf.sigmoid(convolved)
            #output = convolved# ossgan # not work for most f-divergence except for Total Variation and Pearson
	    output = tf.tanh(convolved) # it works
	    #output = tf.nn.relu(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator") as scope:
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator_Unet(a, inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):

        discrim_loss = discrim_loss_sgan((predict_real + EPS), (predict_fake+EPS), a)


    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
	size_image=256
        tv_y_size = size_image
        tv_x_size = size_image

        gen_loss_GAN = gen_loss_sgan(tf.abs(targets-outputs),a)
        gen_loss = gen_loss_GAN * a.gan_weight 
    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply(
        [discrim_loss, gen_loss, gen_loss_GAN ])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step + 1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss=ema.average(gen_loss),
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def main():
    if tf.__version__ != "1.0.0":
        raise Exception("Tensorflow version 1.0.0 required")

    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)
        # remove alpha channel if present
        input_image = input_image[:, :, :3]
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        with tf.variable_scope("generator") as scope:
            batch_output = deprocess(create_generator_Unet(a, preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        output_data = tf.image.encode_png(output_image)
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key": tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        # config = tf.ConfigProto()
        # config.allow_soft_placement = False
        # config.log_device_placement = True
        # config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples(a)
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("gennerator_loss", model.gen_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2 ** 32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            test_start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(a, results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])

                index_path = append_index(a, filesets)
            with open(a.output_dir + '/test.time', 'w') as f:
                f.write(str(time.time() - test_start)+'\n')

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()
            with open(a.output_dir + '/train.precession', 'a') as f:
                f.write('step RFE ACC \n')

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["discrim_loss"] = model.discrim_loss
                    fetches["gen_loss"] = model.gen_loss
                    fetches["gen_loss_GAN"] = model.gen_loss_GAN

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(a, results["display"], step=results["global_step"])
                    append_index(a, filesets, step=True)


                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                        train_epoch, train_step, rate, remaining / 60))
                    print("discrim_loss", results["discrim_loss"])
                    print("gen_loss", results["gen_loss"])
                    print("gen_loss_GAN", results["gen_loss_GAN"])

                #with open(a.output_dir + '/train.log', 'a') as f:
                #    f.write('step gen_loss discrim_loss')
                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)
                    line_str = '%s %s %s\n' % (results["global_step"], results["gen_loss"], results["discrim_loss"])
                    with open(a.output_dir + '/train.log', 'a') as f:
                        f.write(line_str)

                if should(a.test_freq):

                    if a.output_dir is None:
                        raise Exception("checkpoint required for test mode")

                    # load some options from the checkpoint
                    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
                    with open(os.path.join(a.output_dir, "options.json")) as f:
                        for key, val in json.loads(f.read()).items():
                            if key in options:
                                print("loaded", key, "=", val)
                                setattr(a, key, val)
                    # disable these features in test mode
                    a.scale_size = CROP_SIZE
                    a.flip = False

                    # testing
                    # at most, process the test data once
                    max_steps = min(examples.steps_per_epoch, max_steps)

                    for step in range(max_steps):
                        test_results = sess.run(display_fetches)
                        test_filesets = save_images(a, test_results, step=results["global_step"])

                        index_path = append_index(a, test_filesets)

                        if 'ThreeObj_gamma_1.0' in test_filesets[0]['outputs']:
                            #pdb.set_trace()
                            print('file','checkpoints.lr.%s/simulation_test/ossgan_sgan_l1/%s/images/%s'%(a.lr,a.f_type,test_filesets[0]['outputs']))
                            outp = cv2.imread('checkpoints.lr.%s/simulation_test/ossgan_sgan_l1/%s/images/%s'%(a.lr,a.f_type,test_filesets[0]['outputs']))
                            targ = cv2.imread('checkpoints.lr.%s/simulation_test/ossgan_sgan_l1/%s/images/%s' % (
                            a.lr, a.f_type, test_filesets[0]['targets']))
                            inp = cv2.imread('checkpoints.lr.%s/simulation_test/ossgan_sgan_l1/%s/images/%s' % (
                            a.lr, a.f_type, test_filesets[0]['inputs']))
                            rfe = region_fitting_error(outp,targ)
                            acc = calculate_accuracy(inp,outp,targ)
                            #print('region fitting error', rfe)
                            #print('accuracy', acc)
                            line_str = '%s %s %s\n' % (results["global_step"], rfe, acc)
                            with open(a.output_dir + '/train.precession', 'a') as f:
                                f.write(line_str)



                    print("wrote index at", index_path)


                if sv.should_stop():
                    break

            with open(a.output_dir + '/train.time', 'w') as f:
                f.write(str(time.time() - start)+'\n')

main()
