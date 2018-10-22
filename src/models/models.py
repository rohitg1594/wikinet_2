# Model class from string to model name

from src.models.combined.only_prior.linear import Linear
from src.models.combined.only_prior.average import Average
from src.models.combined.only_prior.conv import Conv
from src.models.combined.only_prior.multi_linear import MultiLinear
from src.models.combined.only_prior.position import Position
from src.models.combined.only_prior.rnn import RNN
from src.models.combined.only_prior.with_string import WithString
from src.models.combined.mention_prior import MentionPrior
from src.models.combined.pre_train import PreTrain
from src.models.combined.small_context import SmallContext

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
    pre_train = PreTrain
    small_context = SmallContext

    # Yamada
    yamada_full = YamadaContextStatsString
    yamada_stats = YamadaContextStats
    corpus_vec_average = YamadaCorpusVecAverage
    corpus_vec_linear = YamadaCorpusVecLinear
    corpus_vec_only = YamadaCorpusVecOnly

