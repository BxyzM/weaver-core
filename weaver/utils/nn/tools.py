import numpy as np
import awkward as ak
import tqdm
import time
import torch

from collections import defaultdict, Counter
from .metrics import evaluate_metrics
from ..data.tools import _concat
from ..logger import _logger


def _extract_roc_auc_scalar(metric_results):
    val = metric_results.get('roc_auc_score')
    if val is not None and np.isscalar(val):
        val = float(val)
        if np.isfinite(val):
            return val
        return None

    mat = metric_results.get('roc_auc_score_matrix')
    if mat is None:
        return None

    # OVO AUC matrix is filled only in upper triangle (i < j).
    # Average only valid pairwise entries to avoid zero-padding bias.
    iu = np.triu_indices_from(mat, k=1)
    valid = mat[iu]
    valid = valid[np.isfinite(valid)]
    if valid.size:
        return float(np.mean(valid))
    return None


def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    # print('label', label.shape, label)
    return label


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output

    # preds: (N, num_classes); (N, num_classes, P)
    # label: (N,);             (N, P)
    # mask:  None;             (N, P) / (N, 1, P)
    if preds.ndim > 2:
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    # print('preds', preds.shape, preds)

    if label is not None:
        label = _flatten_label(label, mask)

    return preds, label, mask


def train_classification(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev)
            entry_count += label.shape[0]
            try:
                mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
            except KeyError:
                mask = None
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                logits, label, _ = _flatten_preds(model_output, label=label, mask=mask)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("Acc/train", correct / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, validation_metric='acc', split_name=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                try:
                    mask = y[data_config.label_names[0] + '_mask'].bool().to(dev)
                except KeyError:
                    mask = None
                model_output = model(*inputs)
                logits, label, mask = _flatten_preds(model_output, label=label, mask=mask)
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                if mask is not None:
                    mask = mask.cpu()
                for k, v in y.items():
                    labels[k].append(_flatten_label(v, mask).numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))
                if not for_training and mask is not None:
                    labels_counts.append(np.squeeze(mask.numpy(force=True).sum(axis=-1)))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        if not for_training and split_name:
            tb_mode = f'{tb_mode}_{split_name}'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    if not np.isfinite(scores).all():
        invalid = np.size(scores) - int(np.isfinite(scores).sum())
        _logger.warning(
            'Found %d non-finite score values during evaluation; replacing NaN/inf before metrics.',
            invalid,
        )
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        if scores.ndim == 2 and scores.shape[1] > 1:
            row_sum = scores.sum(axis=1, keepdims=True)
            # Keep valid probability-like rows stable and avoid divide-by-zero.
            np.divide(scores, np.maximum(row_sum, 1e-12), out=scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    
    # Fix for roc_auc_score "y should be a 1d array" error:
    # Convert one-hot encoded labels (N, C) to class indices (N,)
    y_true = labels[data_config.label_names[0]]
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    metric_results = evaluate_metrics(y_true, scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        if not for_training and split_name:
            tb_mode = f'{tb_mode}_{split_name}'
        roc_auc_scalar = _extract_roc_auc_scalar(metric_results)
        if roc_auc_scalar is not None:
            tb_helper.write_scalars([
                ("AUC/%s (epoch)" % tb_mode, roc_auc_scalar, epoch),
            ])

    if for_training:
        if validation_metric is None or validation_metric == 'acc':
            return total_correct / count
        
        # Try to get the specific metric (e.g., roc_auc_score)
        val = metric_results.get(validation_metric)
        if val is not None:
            if np.isscalar(val) and not np.isfinite(float(val)):
                _logger.warning(
                    'Validation metric %s is non-finite (%s). Falling back.',
                    validation_metric, str(val))
            else:
                return val
            
        # Fallback: if user asked for roc_auc_score but it is None/Missing, try to average the matrix
        if validation_metric == 'roc_auc_score':
            auc_scalar = _extract_roc_auc_scalar(metric_results)
            if auc_scalar is not None:
                return auc_scalar

        _logger.warning('Metric %s not found/is None in metric_results. Using accuracy instead.' % validation_metric)
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers


def evaluate_onnx(model_path, test_loader, eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix']):
    import onnxruntime
    sess = onnxruntime.InferenceSession(model_path)

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_correct = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with tqdm.tqdm(test_loader) as tq:
        for X, y, Z in tq:
            # X, y: torch.Tensor; Z: ak.Array
            inputs = {k: v.numpy(force=True) for k, v in X.items()}
            label = y[data_config.label_names[0]].numpy(force=True)
            num_examples = label.shape[0]
            label_counter.update(label)
            score = sess.run([], inputs)[0]
            preds = score.argmax(1)

            scores.append(score)
            for k, v in y.items():
                labels[k].append(v.numpy(force=True))
            for k, v in Z.items():
                observers[k].append(v)

            correct = (preds == label).sum()
            total_correct += correct
            count += num_examples

            tq.set_postfix({
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    
    # Fix for roc_auc_score "y should be a 1d array" error:
    y_true = labels[data_config.label_names[0]]
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=1)

    metric_results = evaluate_metrics(y_true, scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))
    observers = {k: _concat(v) for k, v in observers.items()}
    return total_correct / count, scores, labels, observers


def train_regression(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None):
    model.train()

    data_config = train_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_abs_err = 0
    sum_sqr_err = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].float()
            num_examples = label.shape[0]
            label = label.to(dev)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                model_output = model(*inputs)
                preds = model_output.squeeze()
                loss = loss_func(preds, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            loss = loss.item()

            num_batches += 1
            count += num_examples
            total_loss += loss
            e = preds - label
            abs_err = e.abs().sum().item()
            sum_abs_err += abs_err
            sqr_err = e.square().sum().item()
            sum_sqr_err += sqr_err

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'MSE': '%.5f' % (sqr_err / num_examples),
                'AvgMSE': '%.5f' % (sum_sqr_err / count),
                'MAE': '%.5f' % (abs_err / num_examples),
                'AvgMAE': '%.5f' % (sum_abs_err / count),
            })

            if tb_helper:
                tb_helper.write_scalars([
                    ("Loss/train", loss, tb_helper.batch_train_count + num_batches),
                    ("MSE/train", sqr_err / num_examples, tb_helper.batch_train_count + num_batches),
                    ("MAE/train", abs_err / num_examples, tb_helper.batch_train_count + num_batches),
                ])
                if tb_helper.custom_fn:
                    with torch.no_grad():
                        tb_helper.custom_fn(model_output=model_output, model=model,
                                            epoch=epoch, i_batch=num_batches, mode='train')

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgMSE: %.5f, AvgMAE: %.5f' %
                 (total_loss / num_batches, sum_sqr_err / count, sum_abs_err / count))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("MSE/train (epoch)", sum_sqr_err / count, epoch),
            ("MAE/train (epoch)", sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode='train')
        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_regression(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                        eval_metrics=['mean_squared_error', 'mean_absolute_error', 'median_absolute_error',
                                      'mean_gamma_deviance'],
                        tb_helper=None, validation_metric=None):
    model.eval()

    data_config = test_loader.dataset.config

    total_loss = 0
    num_batches = 0
    sum_sqr_err = 0
    sum_abs_err = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    observers = defaultdict(list)
    start_time = time.time()
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                label = y[data_config.label_names[0]].float()
                num_examples = label.shape[0]
                label = label.to(dev)
                model_output = model(*inputs)
                preds = model_output.squeeze().float()

                scores.append(preds.numpy(force=True))
                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                loss = 0 if loss_func is None else loss_func(preds, label).item()

                num_batches += 1
                count += num_examples
                total_loss += loss * num_examples
                e = preds - label
                abs_err = e.abs().sum().item()
                sum_abs_err += abs_err
                sqr_err = e.square().sum().item()
                sum_sqr_err += sqr_err

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'MSE': '%.5f' % (sqr_err / num_examples),
                    'AvgMSE': '%.5f' % (sum_sqr_err / count),
                    'MAE': '%.5f' % (abs_err / num_examples),
                    'AvgMAE': '%.5f' % (sum_abs_err / count),
                })

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch,
                                                i_batch=num_batches, mode='eval' if for_training else 'test')

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (count, count / time_diff))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("MSE/%s (epoch)" % tb_mode, sum_sqr_err / count, epoch),
            ("MAE/%s (epoch)" % tb_mode, sum_abs_err / count, epoch),
        ])
        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(model_output=model_output, model=model, epoch=epoch, i_batch=-1, mode=tb_mode)

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}
    metric_results = evaluate_metrics(labels[data_config.label_names[0]], scores, eval_metrics=eval_metrics)
    _logger.info('Evaluation metrics: \n%s', '\n'.join(
        ['    - %s: \n%s' % (k, str(v)) for k, v in metric_results.items()]))

    if for_training:
        if validation_metric is None or validation_metric == 'loss':
            return total_loss / count
        if validation_metric in metric_results:
            return metric_results[validation_metric]
        return total_loss / count
    else:
        # convert 2D labels/scores
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_loss / count, scores, labels, observers


class TensorboardHelper(object):

    def __init__(self, tb_comment=None, tb_custom_fn=None, wandb_run=None):
        self.tb_comment = tb_comment
        self.writer = None
        self.wandb_run = wandb_run

        if self.tb_comment is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(comment=self.tb_comment)
            _logger.info('Create Tensorboard summary writer with comment %s' % self.tb_comment)

        # initiate the batch state
        self.batch_train_count = 0

        # load custom function
        self.custom_fn = None
        if tb_custom_fn is not None:
            if self.writer is None:
                _logger.warning('Ignoring tensorboard custom function because tensorboard is disabled.')
            else:
                from weaver.utils.import_tools import import_module
                from functools import partial
                self.custom_fn = import_module(tb_custom_fn, '_custom_fn')
                self.custom_fn = partial(self.custom_fn.get_tensorboard_custom_fn, tb_writer=self.writer)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def write_scalars(self, write_info):
        for tag, scalar_value, global_step in write_info:
            if self.writer is not None:
                self.writer.add_scalar(tag, scalar_value, global_step)
            if self.wandb_run is not None:
                # Do not force W&B `step` from TensorBoard counters because this helper
                # mixes batch-level and epoch-level indices, which can make steps go
                # backwards and cause dropped records.
                payload = {tag: scalar_value}
                if '(epoch)' in tag:
                    payload['epoch'] = global_step
                else:
                    payload['batch_step'] = global_step
                self.wandb_run.log(payload)
