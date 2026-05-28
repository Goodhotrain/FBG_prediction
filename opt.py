import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'paths': [
            dict(name='--root_path',
                 default=".",
                 type=str,
                 help='Global path of root directory'),
            dict(name="--label_path",
               #   default="./data/ID-label.csv",
                 default="./data/ID-label-enhanced-92.csv",
                 type=str,
                 help='Global path of videos', ),
            dict(name="--static_path",
                 default="./data/static.csv",
                 type=str,
                 help='Global path of audios', ),
            dict(name="--nutrition_path",
                 default='./data/nutrition.csv',
                 type=str),
            dict(name='--result_path',
                 type=str,
                 default='./results'),
        ],
        'core': [
            dict(name='--batch_size',
                 default=32,
                 type=int,
                 help='Batch Size'),
            dict(name='--n_classes',
                 default=2,
                 type=int,
                 help='Number of classes'),
            dict(name='--learning_rate',
                 default=1e-4,
                 type=float,
                 help='Initial learning rate',),
            dict(name='--weight_decay',
                 default=0,
                 type=float,
                 help='Weight Decay'),
            dict(name='--test_only',
                 default=False,
                 type=bool,),
        ],
        'common': [
            dict(name='--optimizer',
                 default='sgd',),
            dict(name='--scheduler',
                 default='step',),
            dict(name='--softlabel',
                 default=True,
                 type=bool),
            dict(
                name='--n_threads',
                default = 0,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(
                name='--epochs',
                default=100,
                type=int,
                help='Number of total epochs to run',
            ),
        ]
    }
    
    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args([])
    return args