from argparse import Namespace
from logging import Logger
import os
from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
from torch.optim import Adam
from tqdm import trange

from compose_PyTorch import Compose
from dataloader_PyTorch import DataLoader
from distance_PyTorch import Distance
from evaluate import evaluate, evaluate_predictions
from GlassDataset_PyTorch import GlassDataset
from model import build_model
from nngraph_PyTorch import NNGraph
from predict import predict
from train import train

from chemprop.nn_utils import NoamLR, param_count
from chemprop.parsing import parse_train_args
from chemprop.utils import get_loss_func, get_metric_func, load_checkpoint, save_checkpoint


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """Trains a model and returns test scores on the model checkpoint with the highest validation score"""
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    debug(pformat(vars(args)))

    debug('Loading data')
    data = GlassDataset('metadata/metadata.json', transform=Compose([NNGraph(5), Distance(False)]))
    args.atom_fdim = 3
    args.bond_fdim = args.atom_fdim + 1
    train_data = val_data = test_data = data  # TODO: get separate val/test sets

    train_data_length, test_data_length = len(train_data), len(test_data)

    debug('Total size = {:,} | train size = {:,} | val size = {:,} | test size = {:,}'.format(
        len(data),
        len(train_data),
        len(val_data),
        len(test_data))
    )

    # Set up test set evaluation
    sum_test_preds = np.zeros((test_data_length, args.num_tasks))

    # Convert to iterators
    data = iter(DataLoader(data, batch_size=args.batch_size))
    train_data = val_data = test_data = data  # TODO: get separate val/test sets

    # Get loss and metric functions
    loss_func = get_loss_func(args.dataset_type)
    metric_func = get_metric_func(args.metric)

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, 'model_{}'.format(model_idx))
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            debug('Loading model {} from {}'.format(model_idx, args.checkpoint_paths[model_idx]))
            model, _, _, _ = load_checkpoint(args.checkpoint_paths[model_idx],
                                             current_args=args,
                                             num_tasks=args.num_tasks,
                                             dataset_type=args.dataset_type,
                                             encoder_only=args.load_encoder_only,
                                             logger=logger)
        else:
            debug('Building model {}'.format(model_idx))
            model = build_model(args)

        debug(model)
        debug('Number of parameters = {:,}'.format(param_count(model)))
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(model, None, None, args, os.path.join(save_dir, 'model.pt'))

        # Optimizer and learning rate scheduler
        optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
        scheduler = NoamLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=train_data_length // args.batch_size,
            init_lr=args.init_lr,
            max_lr=args.max_lr,
            final_lr=args.final_lr
        )

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(args.epochs):
            debug('Epoch {}'.format(epoch))

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            val_scores = evaluate(
                model=model,
                data=val_data,
                metric_func=metric_func,
                args=args,
            )

            # Average validation score
            avg_val_score = np.mean(val_scores)
            debug('Validation {} = {:.3f}'.format(args.metric, avg_val_score))
            writer.add_scalar('validation_{}'.format(args.metric), avg_val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(model, None, None, args, os.path.join(save_dir, 'model.pt'))

        # Evaluate on test set using model using model with best validation score
        info('Model {} best validation {} = {:.3f} on epoch {}'.format(model_idx, args.metric, best_score, best_epoch))
        model, _, _, _ = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)
        test_preds = predict(
            model=model,
            data=test_data,
            args=args
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_data.targets(),  # TODO: implement targets
            metric_func=metric_func
        )
        sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.mean(test_scores)
        info('Model {} test {} = {:.3f}'.format(model_idx, args.metric, avg_test_score))
        writer.add_scalar('test_{}'.format(args.metric), avg_test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = sum_test_preds / args.ensemble_size
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds.tolist(),
        targets=test_data.targets(),
        metric_func=metric_func
    )

    # Average ensemble score
    info('Ensemble test {} = {:.3f}'.format(args.metric, np.mean(ensemble_scores)))

    return ensemble_scores


if __name__ == '__main__':
    args = parse_train_args()
    args.num_tasks = 1
    run_training(args)
