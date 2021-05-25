from abc import ABCMeta

from imagemodel.common.dataset_providers.drafter import DrafterP, DrafterT


class RTDrafterT(DrafterT, metaclass=ABCMeta):
    pass


class RTDrafterP(DrafterP, metaclass=ABCMeta):
    pass
