from abc import ABCMeta

from imagemodel.common.dataset_providers.provider import ProviderP, ProviderT


class RTProviderT(ProviderT, metaclass=ABCMeta):
    @property
    def data_description(self):
        return "Reference Tracking Training, Tester provider"


class RTProviderP(ProviderP, metaclass=ABCMeta):
    @property
    def data_description(self):
        return "Reference Tracking Predict provider"
