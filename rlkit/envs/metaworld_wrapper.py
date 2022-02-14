from meta_rand_envs.metaworld import MetaWorldEnv
from . import register_env


@register_env('metaworld')
@register_env('metaworld-ml1-pick-and-place')
@register_env('metaworld-ml1-reach')
@register_env('metaworld-ml1-push')
@register_env('metaworld-ml3')
@register_env('metaworld-ml10')
@register_env('metaworld-ml10-1')
@register_env('metaworld-ml10-2')
@register_env('metaworld-ml10-3')
@register_env('metaworld-ml10-scripted')
@register_env('metaworld-ml10-constrained')
@register_env('metaworld-ml10-constrained-1')
@register_env('metaworld-ml10-constrained-2')
@register_env('metaworld-ml45')
class MetaWorldWrappedEnv(MetaWorldEnv):
    def __init__(self, *args, **kwargs):
        MetaWorldEnv.__init__(self, *args, **kwargs)
