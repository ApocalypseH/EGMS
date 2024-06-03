from .generic import (
    ModelOutput,
)

from .modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqModelOutput,
)

from .dataclasses import (
    dataclass,
)

from .dataset_configuration import (
    DatasetConfig,
)

from .data_utils import (
    MyDataset,
    MyModelDataset,
    collate_fn,
)

from .activations import (
    ACT2FN,
)

from .generation_utils import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor,
    SuppressTokensLogitsProcessor,

    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

from .beam_search import (
    BeamScorer, 
    BeamSearchScorer,
)

import torch
from packaging import version


parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)

is_torch_less_than_1_8 = parsed_torch_version_base < version.parse("1.8.0")

def torch_int_div(tensor1, tensor2):
    """
    A function that performs integer division across different versions of PyTorch.
    """
    if is_torch_less_than_1_8:
        return tensor1 // tensor2
    else:
        return torch.div(tensor1, tensor2, rounding_mode="floor")
