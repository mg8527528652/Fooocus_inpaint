args_parsing = False

def enable_args_parsing(enable=True):
    global args_parsing
    # Always keep args_parsing as False regardless of the input
    # This ensures command line arguments are never parsed
    args_parsing = False
