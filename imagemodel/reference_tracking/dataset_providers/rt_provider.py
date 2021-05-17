from abc import ABCMeta

from imagemodel.common.dataset_providers.provider import ProviderP, ProviderT


class RTProviderT(ProviderT, metaclass=ABCMeta):
    pass


class RTProviderP(ProviderP, metaclass=ABCMeta):
    pass
