from argparse import ArgumentParser, RawTextHelpFormatter


class Parser:
    def __init__(self, parser: ArgumentParser):
        self.parser = parser

    @classmethod
    def init_from_scratch(cls, description: str):
        parser = ArgumentParser(
            description=description,
            formatter_class=RawTextHelpFormatter,
        )
        return cls(parser=parser)

    def get_str_from_command(self, sys_argv):
        command_str = list(map(lambda el: str(el), sys_argv))
        list_to_command_str = " ".join(map(str, command_str))
        parsed_args_str = ""
        for arg in list(vars(args))[:-1]:
            parsed_args_str += "- {}: {}\n".format(arg, getattr(args, arg))
        parsed_args_str += "- {}: {}".format(
            list(vars(args))[-1], getattr(args, list(vars(args))[-1])
        )


class TrainingParser(Parser):
    def __init__(self, parser: ArgumentParser):
        super().__init__(parser)

    def add_training_dataset(self):
        self.parser.add_argument(
            "--training_dataset",
            type=str,
            help="Should be one of models in `Datasets` "
                 "of `imagemodel/binary_segmentations/configs/datasets.py`. \n"
                 "ex) 'oxford_iiit_pet_v3_training'",
        )
