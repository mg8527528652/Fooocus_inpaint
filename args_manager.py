import ldm_patched.modules.args_parser as args_parser
import argparse

# Define args as a module-level variable
args = None

def parse_args():
    global args
    # Create a new parser for the application-specific arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

    parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
    parser.add_argument("--disable-preset-selection", action='store_true',
                                    help="Disables preset selection in Gradio.")

    parser.add_argument("--language", type=str, default='default',
                                    help="Translate UI using json files in [language] folder. "
                                    "For example, [--language example] will use [language/example.json] for translation.")

    # For example, https://github.com/lllyasviel/Fooocus/issues/849
    parser.add_argument("--disable-offload-from-vram", default=False, action="store_true",
                                    help="Force loading models to vram when the unload can be avoided. "
                                    "Some Mac users may need this.")

    parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
    parser.add_argument("--disable-image-log", action='store_true',
                                    help="Prevent writing images and logs to the outputs folder.")

    parser.add_argument("--disable-analytics", action='store_true',
                                    help="Disables analytics for Gradio.")

    parser.add_argument("--disable-metadata", action='store_true',
                                    help="Disables saving metadata to images.")

    parser.add_argument("--disable-preset-download", action='store_true',
                                    help="Disables downloading models for presets", default=False)

    parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                    help="Disables enhance output sorting for final image gallery.")

    parser.add_argument("--enable-auto-describe-image", action='store_true',
                                    help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

    parser.add_argument("--always-download-new-model", action='store_true',
                                    help="Always download newer models", default=False)

    parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                    type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

    # Set default values for certain arguments
    parser.set_defaults(
        disable_cuda_malloc=True,
        in_browser=True,
        port=None
    )

    # Parse the arguments without actually processing command line (for Ray compatibility)
    parsed_args = parser.parse_args([])

    # Get the base args from args_parser
    base_args = args_parser.args
    
    # Combine the attributes from base_args into our args
    for attr in dir(base_args):
        if not attr.startswith('_'):  # Skip private attributes
            if not hasattr(parsed_args, attr):  # Don't override if already set
                setattr(parsed_args, attr, getattr(base_args, attr))

    # (Disable by default because of issues like https://github.com/lllyasviel/Fooocus/issues/724)
    parsed_args.always_offload_from_vram = False

    if parsed_args.disable_analytics:
        import os
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

    if parsed_args.disable_in_browser:
        parsed_args.in_browser = False

    # Set module-level args variable for other modules to access
    args = parsed_args
    return parsed_args

# Initialize args right away to ensure it's available for imports
args = parse_args()


