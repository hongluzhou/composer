import os
import pdb

import torch
import torch.nn as nn


def create_model(args, logger=None):
    model = None

    if args.model_type == 'composer':
        from models.composer import COMPOSER
        model = COMPOSER(args)
    else:
        raise ValueError("Model {} not recognized.".format(
            args.model_type))

    if logger:
        logger.info("--> model {} was created".format(
            args.model_type))

    return model
