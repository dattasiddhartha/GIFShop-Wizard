import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import flags, app
from absl.flags import FLAGS
from vision.foreground_removal.pix2pix.utils.model import Pix2Pix
from vision.foreground_removal.pix2pix.utils.dataset import train_pipeline, test_pipeline

flags.DEFINE_integer('buffer_size', 400, 'size of buffer')
flags.DEFINE_integer('batch_size', 4, 'size of batch')
flags.DEFINE_integer('width', 256, 'width of resulting images')
flags.DEFINE_integer('height', 256, 'height of resulting images')
flags.DEFINE_float('lambda_p', 100, 'lambda parameter')
flags.DEFINE_integer('epochs', 25, 'Number of epochs to train from', short_name='e')
flags.DEFINE_string('checkpoint', 'pix2pix/checkpoint/', 'Checkpoint directory')
flags.DEFINE_string('training_dir', 'input/train/', 'Path for training samples', short_name='train')
flags.DEFINE_string('testing_dir', 'input/test/', 'Path for testing samples', short_name='test')
flags.DEFINE_bool('restore_check', True, 'Restore last checkpoint in folder --checkpoint', short_name='restore')
flags.DEFINE_integer('num_images', -1, 'Number of images to take from dataset', short_name='n')
flags.DEFINE_integer('test_samples', 2, 'Number of generated samples for testing')
flags.DEFINE_string('mode', 'test', 'Mode: train or test')

def main(_argv):
    print('Parameters:\n')
    print(f'Image = [{FLAGS.height}x{FLAGS.width}]\n')
    print(f'Lambda = {FLAGS.lambda_p}\n')
    print(f'Number of images = {FLAGS.num_images}\n')

    mode = FLAGS.mode
    if mode == 'train':
        train_dataset = train_pipeline(FLAGS.training_dir, FLAGS.buffer_size, FLAGS.width, FLAGS.height, FLAGS.num_images,
                                       FLAGS.batch_size)
        test_dataset = test_pipeline(FLAGS.testing_dir, FLAGS.width, FLAGS.height, FLAGS.num_images)

        p2p = Pix2Pix(mode, train_dataset, test_dataset, FLAGS.lambda_p, FLAGS.epochs, FLAGS.checkpoint, FLAGS.restore_check,
                      FLAGS.test_samples)
        p2p.fit()
    elif mode == 'test':
        train_dataset = []
        test_dataset = test_pipeline(FLAGS.testing_dir, FLAGS.width, FLAGS.height, FLAGS.num_images)

        p2p = Pix2Pix(mode, train_dataset, test_dataset, FLAGS.lambda_p, FLAGS.epochs, FLAGS.checkpoint, FLAGS.restore_check,
                      FLAGS.test_samples)

        p2p.test_model()


if __name__ == '__main__':
    try:
        app.run(main)
    except Exception as e:
        print(f'Error: {e}')
