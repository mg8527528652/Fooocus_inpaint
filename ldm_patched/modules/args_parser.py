import argparse
import enum
import ldm_patched.modules.options

class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        choices = tuple(e.value for e in enum_type)
        kwargs.setdefault("choices", choices)
        kwargs.setdefault("metavar", f"[{','.join(list(choices))}]")

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)

# Define LatentPreviewMethod at the module level so it can be imported
class LatentPreviewMethod(enum.Enum):
    NoPreviews = "none"
    Auto = "auto"
    Latent2RGB = "fast"
    TAESD = "taesd"

# Create parser as a module-level variable so it can be accessed by args_manager.py
parser = argparse.ArgumentParser()

def parse_args():
    global parser
    # Reset parser to ensure clean state
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--listen", type=str, default="127.0.0.1", metavar="IP", nargs="?", const="0.0.0.0")
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--disable-header-check", type=str, default=None, metavar="ORIGIN", nargs="?", const="*")
    parser.add_argument("--web-upload-size", type=float, default=100)
    parser.add_argument("--hf-mirror", type=str, default=None)

    parser.add_argument("--external-working-path", type=str, default=None, metavar="PATH", nargs='+', action='append')
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--temp-path", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--in-browser", action="store_true")
    parser.add_argument("--disable-in-browser", action="store_true")
    parser.add_argument("--gpu-device-id", type=int,  default=0, metavar="DEVICE_ID")
    cm_group = parser.add_mutually_exclusive_group()
    cm_group.add_argument("--async-cuda-allocation", action="store_true")
    cm_group.add_argument("--disable-async-cuda-allocation", action="store_true")

    parser.add_argument("--disable-attention-upcast", action="store_true")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--all-in-fp32", action="store_true")
    fp_group.add_argument("--all-in-fp16", action="store_true")

    fpunet_group = parser.add_mutually_exclusive_group()
    fpunet_group.add_argument("--unet-in-bf16", action="store_true")
    fpunet_group.add_argument("--unet-in-fp16", action="store_true")
    fpunet_group.add_argument("--unet-in-fp8-e4m3fn", action="store_true")
    fpunet_group.add_argument("--unet-in-fp8-e5m2", action="store_true")

    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--vae-in-fp16", action="store_true")
    fpvae_group.add_argument("--vae-in-fp32", action="store_true")
    fpvae_group.add_argument("--vae-in-bf16", action="store_true")

    parser.add_argument("--vae-in-cpu", action="store_true")

    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
    fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
    fpte_group.add_argument("--clip-in-fp16", action="store_true")
    fpte_group.add_argument("--clip-in-fp32", action="store_true")


    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)

    parser.add_argument("--disable-ipex-hijack", action="store_true")

    # Use the module-level LatentPreviewMethod class
    parser.add_argument("--preview-option", type=LatentPreviewMethod, default=LatentPreviewMethod.NoPreviews, action=EnumAction)

    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--attention-split", action="store_true")
    attn_group.add_argument("--attention-quad", action="store_true")
    attn_group.add_argument("--attention-pytorch", action="store_true")

    parser.add_argument("--disable-xformers", action="store_true")

    vram_group = parser.add_mutually_exclusive_group()
    vram_group.add_argument("--always-gpu", default=True)
    vram_group.add_argument("--always-high-vram", default=True)
    vram_group.add_argument("--always-normal-vram", default=False)
    vram_group.add_argument("--always-low-vram", default=False)
    vram_group.add_argument("--always-no-vram", default=False)
    vram_group.add_argument("--always-cpu", type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

    parser.add_argument("--always-offload-from-vram", default=False)
    parser.add_argument("--pytorch-deterministic", action="store_true")

    parser.add_argument("--disable-server-log", default=False)
    parser.add_argument("--debug-mode", default=False)
    parser.add_argument("--is-windows-embedded-python", default=False)

    parser.add_argument("--disable-server-info", default = False)

    parser.add_argument("--multi-user", action="store_true")

    # Always use the empty list for args regardless of ldm_patched.modules.options.args_parsing
    # This ensures no command line args are ever parsed
    args = parser.parse_args([])

    if args.is_windows_embedded_python:
        args.in_browser = True

    if args.disable_in_browser:
        args.in_browser = False

    return args

args = parse_args()