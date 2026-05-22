from __future__ import annotations

import string

import numpy as np


class CharTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, max_length: int = 96) -> None:
        self.max_length = max_length
        charset = string.ascii_lowercase + string.digits + " .,;:!?+-_/()[]"
        self.id_to_token = [self.PAD, self.BOS, self.EOS, self.UNK] + list(dict.fromkeys(charset))
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        text = text.lower()
        ids = [self.token_to_id[self.BOS]]
        for char in text:
            ids.append(self.token_to_id.get(char, self.token_to_id[self.UNK]))
        ids.append(self.token_to_id[self.EOS])
        ids = ids[: self.max_length]
        attention = np.ones((len(ids),), dtype=np.float32)

        if len(ids) < self.max_length:
            pad_count = self.max_length - len(ids)
            ids += [self.token_to_id[self.PAD]] * pad_count
            attention = np.pad(attention, (0, pad_count), mode="constant", constant_values=0.0)

        return np.asarray(ids, dtype=np.int64), attention.astype(np.float32)
