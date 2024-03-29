# Model class from string to model name

from src.models.combined.only_prior.linear import Linear
from src.models.combined.only_prior.average import Average
from src.models.combined.only_prior.conv import Conv
from src.models.combined.only_prior.multi_linear import MultiLinear
from src.models.combined.only_prior.position import Position
from src.models.combined.only_prior.rnn import RNN
from src.models.combined.only_prior.with_string import WithString
from src.models.combined.mention_prior import MentionPrior
from src.models.combined.pre_train_context import PreTrainContext
from src.models.combined.small_context import SmallContext

from src.models.combined.full_context import FullContext
from src.models.combined.full_context_attention import FullContextAttn
from src.models.combined.full_context_string_scalar import FullContextStringScalar
from src.models.combined.full_context_string_combined import FullContextStringCombined
from src.models.combined.full_context_string_linear_scalar import FullContextStringLinearScalar
from src.models.combined.full_context_string_from_scratch_ent import FullContextStringFromScratchEnt
from src.models.combined.full_context_string_per_param_weight import FullContextStringPerParamWeight
from src.models.combined.full_context_string_dan import FullContextStringDan
from src.models.combined.full_context_string_train_ent import FullContextStringTrainEnt

from src.models.yamada.yamada_context_stats_string import YamadaContextStatsString
from src.models.yamada.yamada_context_stats import YamadaContextStats
from src.models.yamada.corpus_vec_average import YamadaCorpusVecAverage
from src.models.yamada.corpus_vec_only import YamadaCorpusVecOnly
from src.models.yamada.corpus_vec_linear import YamadaCorpusVecLinear


class Models:

    # Combined
    linear = Linear
    average = Average
    conv = Conv
    multi_linear = MultiLinear
    position = Position
    rnn = RNN
    with_string = WithString
    mention_prior = MentionPrior
    pre_train_context = PreTrainContext
    small_context = SmallContext

    full_context = FullContext
    full_context_attention = FullContextAttn
    full_context_string_scalar = FullContextStringScalar
    full_context_string_combined = FullContextStringCombined
    full_context_string_linear_scalar = FullContextStringLinearScalar
    full_context_string_from_scratch_ent = FullContextStringFromScratchEnt
    full_context_string_per_param_weight = FullContextStringPerParamWeight
    full_context_string_dan = FullContextStringDan
    full_context_string_train_ent = FullContextStringTrainEnt

    # Yamada
    yamada_full = YamadaContextStatsString
    yamada_stats = YamadaContextStats
    corpus_vec_average = YamadaCorpusVecAverage
    corpus_vec_linear = YamadaCorpusVecLinear
    corpus_vec_only = YamadaCorpusVecOnly

