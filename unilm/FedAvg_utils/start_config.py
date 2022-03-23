import os

def print_options(args, model):
    message = ''

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = num_params / 1000000

    message += "================ FL train of %s with total model parameters: %2.1fM  ================\n" % (args.model, num_params)

    message += '++++++++++++++++ Other Train related parameters ++++++++++++++++ \n'
    
    for k, v in sorted(vars(args).items()):
        comment = ''
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '++++++++++++++++  End of show parameters ++++++++++++++++ '

    ## save to disk of current log
    
    args.file_name = os.path.join(args.output_dir, 'log_file.txt')
    
    with open(args.file_name, 'wt') as args_file:
        args_file.write(message)
        args_file.write('\n')
    
    print(message)