def text_to_args(args):
    """convert text to args, in order to create Params
    """
    args = [arg.rstrip() for arg in args.split('--') if arg != '']
    result_arg = []
    for arg in args:
        texts = arg.split(' ')
        result_arg.append('--' + texts[0])
        for i in range(1, len(texts)):
            result_arg.append(texts[i])
    return result_arg