from .masked_editor import MaskedMotionEditor
from .prefix_future_decoder import PrefixFutureDecoder


def create_model(
    model_arch: str,
    pose_dim: int,
    edit_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.1,
    text_vocab_size: int = 0,
    max_frames: int = 60,
    base_step_scale: float = 0.01,
    active_step_scale: float = 0.05,
):
    if model_arch == 'masked_editor':
        return MaskedMotionEditor(
            pose_dim=pose_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            text_vocab_size=text_vocab_size,
            max_frames=max_frames,
        )
    if model_arch == 'prefix_future_decoder':
        return PrefixFutureDecoder(
            pose_dim=pose_dim,
            edit_dim=edit_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            text_vocab_size=text_vocab_size,
            max_frames=max_frames,
            base_step_scale=base_step_scale,
            active_step_scale=active_step_scale,
        )
    raise ValueError(f'Unsupported model_arch={model_arch}')


__all__ = ['MaskedMotionEditor', 'PrefixFutureDecoder', 'create_model']
