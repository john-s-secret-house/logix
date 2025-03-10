from scalablerunner.taskrunner import TaskRunner

if __name__ == '__main__':
    config = {
        'Section: Training CIFAR10 CLF RATIO, Save Indice, Loss, and Logits': { # Each section would be executed sequentially.
            'GTX A800': {
                'Call': 'python compute_influences.py',
                'Param': {
                    '--ckpt_path': ['../../../ckpts/CIFAR2_32_bs2048_ckpts'],
                    '--data': ['CIFAR2_32'],
                    '--md_num': [1, 2, 3, 5, 10, 6, 7, 8, 9],
                    # '--batch_size': [2048],
                    # '--damping': [None],
                    '--lora': ['pca', 'random', 'none'],
                    '--hessian': ['kfac', 'none', 'ekfac'],
                    '--save': ['grad'],
                },
                'Async': { # The task under the same group would be schedule to the resources by TaskRunner during runtime.
                    # '--device': ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
                    '--device': ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4']
                }
            },
        },
    }

    tr = TaskRunner(config=config)
    tr.run()