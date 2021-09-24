import os
import time
import numpy as np

from huitre.evaluators import Evaluator
from huitre.logging import get_logger


class Trainer:
    """
    Trainer is responsible to estimate paramaters for a given model
    """
    def __init__(self, sess, model, params):
        """
        Initialization a trainer. The trainer will be responsible
        to train the model.
        :param sess: global session
        :param model: model to be trained
        :param params: hyperparameters for training
        """
        self.sess = sess
        self.model = model
        self.params = params
        self.model_dir = params['training']['model_dir']
        self.n_epochs = self.params['training'].get('num_epochs', 20)
        self.logger = get_logger()

    def fit(self, train_generator, valid_generator,
            train_n_batches_per_epoch, valid_n_batches_per_epoch):
        """
        Training model
        :param train_generator:
        :param valid_generator:
        :param train_n_batches_per_epoch:
        :param valid_n_batches_per_epoch:
        :return:
        """
        metrics_path = '{}/metrics.csv'.format(self.model_dir)
        if os.path.isfile(metrics_path):
            os.remove(metrics_path)

        # create an evaluator
        evaluator = Evaluator(
            self.params['eval'],
            ref_user_items=self.params['training']['ref_user_items']['valid'])

        # training epoch by epoch
        best_valid_loss = 1e100
        best_ep = -1
        with open(metrics_path, 'w') as f:
            header = 'epoch,lr,train_loss,val_loss,' + evaluator.metric_str()
            f.write(f'{header}\n')
            # for each epoch
            for ep in range(1, self.n_epochs + 1):
                start_time = time.time()
                # calculate train loss
                train_loss = self._get_epoch_loss(
                    train_generator, ep,
                    n_batches=train_n_batches_per_epoch,
                    mode='train')
                # calculate validation loss
                valid_loss = self._get_epoch_loss(
                    valid_generator, ep,
                    n_batches=valid_n_batches_per_epoch,
                    mode='valid')
                if valid_loss < best_valid_loss or ep == 1:
                    save_model = True
                    best_valid_loss = valid_loss
                    best_ep = ep
                else:
                    save_model = False
                logged_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, valid_loss, start_time)
                self.logger.info(', '.join(logged_message))
                metric_message = self._get_message(
                    ep, self.model.learning_rate,
                    train_loss, valid_loss, start_time,
                    logged=False)
                f.write(','.join(metric_message) + '\n')
                f.flush()
                if save_model:
                    save_path = f'{self.model_dir}/' \
                                f'{self.model.__class__.__name__.lower()}' \
                                f'-epoch_{ep}'
                    self.model.save(save_path=save_path, global_step=ep)
            self.logger.info(f'Best validation loss: {best_valid_loss}, '
                             f'on epoch {best_ep}')

    def _get_epoch_loss(self, batch_generator, epoch_id,
                        n_batches, mode='train'):
        """
        Forward pass for an epoch
        :param batch_generator:
        :param epoch_id:
        :param n_batches:
        :param mode:
        :return:
        """
        losses = []
        reg_losses = []
        desc = f'Optimizing epoch #{epoch_id}' if mode == 'train' \
            else f'Calculating validation loss for epoch {epoch_id}'
        self.logger.info(desc)
        # for each batch
        for step in range(1, n_batches):
            # get batch data
            batch_data = batch_generator.next_batch()
            batch_loss, batch_reg = self._get_batch_loss(batch=batch_data,
                                                         mode=mode)
            if not np.isinf(batch_loss) and not np.isnan(batch_loss):
                losses.append(batch_loss)
            if not np.isinf(batch_reg) and not np.isnan(batch_reg):
                reg_losses.append(batch_reg)
            if mode == 'train':
                if step % self.params['logs']['log_freq'] == 0:
                    message = [f'Step #{step}',
                               f'Train loss {np.mean(losses, axis=0)}']
                    reg = np.mean(reg_losses, axis=0)
                    if reg > 0:
                        message.append(f'Regularization {reg}')
                    self.logger.info(': '.join(message))
        loss = np.mean(losses, axis=0)
        return loss

    def _get_batch_loss(self, batch, mode='train'):
        """
        Forward pass for a batch
        :param batch:
        :param mode:
        :return:
        """
        feed_dict = self.model.build_feedict(batch)
        reg_loss = 0
        if mode == 'train':
            if self.model.reg_loss is None:
                _, loss = self.sess.run([self.model.train_ops, self.model.loss],
                                        feed_dict=feed_dict)
            else:
                _, loss, reg_loss = self.sess.run(
                    [self.model.train_ops, self.model.loss, self.model.reg_loss],
                    feed_dict=feed_dict)
        else:
            loss = self.sess.run(self.model.loss,
                                 feed_dict=feed_dict)
        return loss, reg_loss

    def _get_reco_items(self, num_items, num_users=-1):
        train_user_items = self.params['training']['ref_user_items']['train']
        valid_user_items = self.params['training']['ref_user_items']['valid']

        if 0 < num_users < len(valid_user_items):
            # get random user_ids array
            valid_user_ids = np.random.choice(
                list(valid_user_items.keys()),
                size=num_users,
                replace=False)
        else:
            valid_user_ids = valid_user_items.keys()

        # get recommend items
        reco_items = self.model.recommend(
            users=valid_user_ids,
            excluded_items=train_user_items,
            num_items=num_items,
            n_users_in_chunk=100
        )
        # remove users that do not have any interactions in validation
        reco_items = {uid: u_reco for uid, u_reco in reco_items.items()
                      if len(valid_user_items[uid]) > 0}
        return reco_items

    @classmethod
    def _get_message(cls, ep, learning_rate,
                     train_loss, valid_loss, start_time, logged=True):
        duration = int(time.time() - start_time)
        ss, duration = duration % 60, duration // 60
        mm, hh = duration % 60, duration // 60
        if logged is True:
            message = [f'Epoch #{ep}:',
                       f'Learning Rate {learning_rate:9.7f}',
                       f'Train loss {train_loss:8.5f}',
                       f'Validation loss {valid_loss:8.5f}',
                       f'Period:{hh:0>2d}h{mm:0>2d}m{ss:0>2d}s']
        else:
            message = [f'{ep}:',
                       f'{learning_rate:9.7f}',
                       f'{train_loss:8.5f}',
                       f'{valid_loss:8.5f}']
        return message
