import os


def return_dataset(dataset_name, args, train_model=True):
    if dataset_name == 'volleyball':
        from datasets.volleyball import Volleyball
        if train_model:
            return Volleyball(args, 'train')
        else:
            return Volleyball(args, 'test', print_cls_idx=True)
        
    elif dataset_name == 'collective':
        from datasets.collective import Collective
        if train_model:
            return Collective(args, 'train')
        else:
            return Collective(args, 'test', print_cls_idx=True)
        
    else:
        print('Please check the dataset name!')
        os._exit(0)

